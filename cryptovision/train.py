"""
Train CryptoVision models with the two-stage protocol of the paper, driven by a YAML config.

  stage 1   backbone frozen, train shared layers and taxonomic heads   (config: training)
  stage 2   top backbone layers unfrozen, fine-tune                    (config: finetune)
  then      test-set metrics as in Table 1 (balanced accuracy, weighted precision /
            recall / F1 on the joint family-genus-species label) and TAS

Usage (from the repository root):
  python -m cryptovision.train --catalog my_images/                 # folder: family/genus/species/
  python -m cryptovision.train --catalog my_images/catalog.csv --arch all
  python -m cryptovision.train --config my_config.yml

--catalog takes a catalog CSV or an image folder (its catalog.csv is created on first use,
see cryptovision.data.make_catalog).

Each run writes to <output_dir>/<arch>_<timestamp>/: the resolved config, training
histories, weights, test metrics and per-image test predictions. Console output is
also saved to <output_dir>/logs/.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

ARCHS = ["std", "concat", "gated", "att"]
LEVELS = ["family", "genus", "species"]
DEFAULT_CONFIG = Path(__file__).parent / "settings.yml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--arch", choices=ARCHS + ["all"], help="overrides model.architecture")
    p.add_argument("--catalog", help="catalog CSV or image folder; overrides data.catalog")
    p.add_argument("--out-dir", type=Path, help="overrides general.output_dir")
    p.add_argument("--max-per-species", type=int, help="subsample each species (smoke tests only)")
    return p.parse_args()


def load_config(args):
    cfg = yaml.safe_load(args.config.read_text())
    if args.arch:
        cfg["model"]["architecture"] = args.arch
    if args.catalog:
        cfg["data"]["catalog"] = args.catalog
    if args.out_dir:
        cfg["general"]["output_dir"] = str(args.out_dir)
    catalogs = cfg["data"]["catalog"]
    catalogs = [catalogs] if isinstance(catalogs, str) else catalogs
    cfg["data"]["catalog"] = [os.path.expandvars(c) for c in catalogs]
    if any("$" in c for c in cfg["data"]["catalog"]):
        sys.exit("No data set: pass --catalog or set CRYPTOVISION_DATA.")
    return cfg


class _Tee:
    """Write a stream to the console and a log file at the same time."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(out_dir):
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    log = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log)
    print(f"Logging to {path}")
    print("Command:", " ".join(sys.argv))


def setup_tf(general):
    random.seed(general["seed"])
    np.random.seed(general["seed"])
    # Hide TF's C++ info/warnings and the harmless ptxas errors on Windows; Python errors still raise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")  # "Using a while_loop for converting ..." warnings
    tf.random.set_seed(general["seed"])
    # CUDA has no deterministic AdjustContrastv2 (RandomContrast): GPU training fails with it on
    if general.get("deterministic"):
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    if general.get("mixed_precision"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    return tf


def load_splits(cfg, max_per_species=None):
    from cryptovision.data import load_catalogs, quality_control, split_data

    d = cfg["data"]
    data = quality_control(
        load_catalogs(d["catalog"]),
        min_samples_per_class=d["min_samples_per_class"],
        remove_duplicates=d["remove_duplicates"],
        stratify_by=d["stratify_by"],
    )
    missing = [p for p in data["image_path"].head(50) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/50 sampled image paths do not exist, e.g. {missing[0]!r}. "
            "Check the catalog's image_path column."
        )

    train, val, test = split_data(
        data,
        test_fraction=d["test_fraction"],
        val_fraction=d["val_fraction"],
        seed=cfg["general"]["seed"],
        stratify_by=d["stratify_by"],
    )
    if max_per_species:
        train, val, test = (
            df.groupby("species", group_keys=False).head(max_per_species) for df in (train, val, test)
        )
    train, val, test = (df.reset_index(drop=True) for df in (train, val, test))
    counts = data["species"].value_counts()
    counts_path = Path(cfg["general"]["output_dir"]) / "species_counts.csv"
    counts.rename_axis("species").rename("images").to_csv(counts_path)
    print(
        f"Images per species: min {counts.min()} ({counts.idxmin()}), median {int(counts.median())}, "
        f"max {counts.max()} ({counts.idxmax()}); full list in {counts_path}"
    )
    print(
        f"Dataset: {len(data)} images | {data['family'].nunique()} families, "
        f"{data['genus'].nunique()} genera, {data['species'].nunique()} species | "
        f"train {len(train)}, val {len(val)}, test {len(test)}"
    )
    return train, val, test


def build_model(tf, cfg, arch, n_classes, parent_genus, parent_species):
    from cryptovision import losses
    from cryptovision.models import CryptoVisionModels as cv_models
    from cryptovision.weights import build_model

    m, aug = cfg["model"], cfg["model"]["augmentation"]

    class CPUAugmentation(tf.keras.layers.Layer):
        """Run the wrapped augmentation layers on the CPU."""

        def __init__(self, augmentation, **kwargs):
            super().__init__(**kwargs)
            self.augmentation = augmentation

        def call(self, inputs, training=None):
            with tf.device("/CPU:0"):
                return self.augmentation(inputs, training=training)

    # Keras 2 (TF 2.10, native Windows): build augmentation in float32 (some ops
    # have no float16 kernel) and run it on the CPU, because TF 2.10 has no
    # prebuilt GPU kernels for newer GPUs (e.g. RTX Ada, sm_89) for some of its
    # ops and fails to JIT-compile them ("JIT compilation failed" in RandomCrop)
    policy = tf.keras.mixed_precision.global_policy()
    if losses._KERAS2:
        tf.keras.mixed_precision.set_global_policy("float32")
    augmentation = cv_models.augmentation_layer(
        seed=cfg["general"]["seed"],
        zoom_factor=tuple(aug["zoom"]),
        contrast=aug["contrast"],
        brightness=aug["brightness"],
        rotation=aug["rotation"],
        translation=aug["translation"],
        image_size=m["image_size"],
        gauss_noise=aug["gaussian_noise"],
    )
    if losses._KERAS2:
        augmentation = CPUAugmentation(augmentation, name="cpu_augmentation")
    tf.keras.mixed_precision.set_global_policy(policy)

    base = build_model(m, arch, n_classes, augmentation=augmentation)
    return losses.TFCLModel(
        parent_genus,
        parent_species,
        lambda_gen=cfg["loss"]["lambda_genus"],
        lambda_spe=cfg["loss"]["lambda_species"],
        inputs=base.inputs,
        outputs=base.outputs,
        name=f"CVis_{arch}_tfcl",
    )


def compile_model(tf, cfg, model, stage):
    from cryptovision import losses

    optimizers = {"adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop}
    lc = cfg["loss"]
    focal = losses.focal_loss(alpha=lc["focal_alpha"], gamma=lc["focal_gamma"])
    model.compile(
        optimizer=optimizers[stage["optimizer"].lower()](stage["learning_rate"]),
        loss={lvl: focal for lvl in LEVELS},
        metrics={lvl: ["accuracy"] for lvl in LEVELS},
        loss_weights=dict(zip(LEVELS, lc["loss_weights"])),
    )


def callbacks(tf, stage, csv_path):
    es, rlr = stage["early_stopping"], stage["reduce_lr_on_plateau"]
    cbs = [
        epoch_logger(tf, stage["epochs"]),  # first: reads the LR before ReduceLROnPlateau changes it
        tf.keras.callbacks.EarlyStopping(
            monitor=es["monitor"], patience=es["patience"],
            restore_best_weights=es["restore_best_weights"], verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=rlr["monitor"], factor=rlr["factor"], patience=rlr["patience"], min_lr=rlr["min_lr"],
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path)),
    ]
    if stage.get("lr_schedule"):
        factor, drops = stage["lr_schedule"]["factor"], stage["lr_schedule"]["epochs"]
        cbs.append(tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * factor if epoch in drops else lr
        ))
    return cbs


def epoch_logger(tf, epochs):
    """A progress bar during each epoch (console only) and one summary line per epoch."""
    console = sys.__stdout__  # bypass the log file: the bar redraws itself with \r
    live = console.isatty()

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            print(f"{'':12}{'train':>30}   {'validation':>30}")
            print(f"{'epoch':<8}{'loss':>8}  {'acc: fam  gen  spe':>20}   "
                  f"{'loss':>8}  {'acc: fam  gen  spe':>20}   {'lr':>7}  {'time':>7}")

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch, self.start = epoch + 1, time.time()

        def on_train_batch_end(self, batch, logs=None):
            if not live:
                return
            steps, done = self.params.get("steps"), batch + 1
            if steps:
                filled = 25 * done // steps
                eta = (time.time() - self.start) / done * (steps - done)
                bar = f"[{'=' * filled}{'.' * (25 - filled)}] {done}/{steps}  ETA {_hms(eta)}"
            else:
                bar = f"{done} batches"
            console.write(f"\r{self.epoch}/{epochs}  {bar}  loss {logs['loss']:.4f}   ")
            console.flush()

        def on_epoch_end(self, epoch, logs=None):
            if live:
                console.write("\r" + " " * 79 + "\r")
            acc = lambda prefix: "  ".join(f"{logs[f'{prefix}{lvl}_accuracy']:.3f}" for lvl in LEVELS)
            lr = float(np.array(self.model.optimizer.learning_rate))
            print(f"{f'{self.epoch}/{epochs}':<8}{logs['loss']:>8.4f}  {acc(''):>20}   "
                  f"{logs['val_loss']:>8.4f}  {acc('val_'):>20}   {lr:>7.1e}  "
                  f"{_hms(time.time() - self.start):>7}")

    return EpochLogger()


def _hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def evaluate(model, tf_test, df_test, class_names, parent_genus, parent_species):
    """Test-set metrics as reported in Table 1, plus per-image predictions."""
    probs = model.predict(tf_test, verbose=0)
    preds = df_test[[c for c in ["image_path", "source"] + LEVELS if c in df_test]].copy()
    idx = {}
    for lvl, p in zip(LEVELS, probs):
        idx[lvl] = np.argmax(p, axis=1)
        preds[f"{lvl}_pred"] = np.array(class_names[lvl])[idx[lvl]]
        preds[f"{lvl}_conf"] = np.max(p, axis=1).astype(float)

    def metrics(y_true, y_pred):
        return {
            "accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }

    results = {}
    for lvl in LEVELS:
        results[lvl] = metrics(preds[lvl].astype(str), preds[f"{lvl}_pred"].astype(str))
        results[lvl]["mean_confidence"] = round(float(preds[f"{lvl}_conf"].mean()), 4)
        results[lvl]["std_confidence"] = round(float(preds[f"{lvl}_conf"].std(ddof=0)), 4)

    true = preds["family"] + " " + preds["genus"] + " " + preds["species"]
    pred = preds["family_pred"] + " " + preds["genus_pred"] + " " + preds["species_pred"]
    results["overall"] = metrics(true, pred)
    pg, ps = np.array(parent_genus), np.array(parent_species)
    aligned = (pg[idx["genus"]] == idx["family"]) & (ps[idx["species"]] == idx["genus"])
    results["overall"]["tas"] = round(float(np.mean(aligned)), 4)
    return results, preds


def run(tf, cfg, arch, splits, out_root):
    from cryptovision.data import make_parent_lists, tensorflow_dataset
    from cryptovision.weights import unfreeze_top_layers

    df_train, _, df_test = splits
    size = cfg["model"]["image_size"]
    tf_ds = {
        name: tensorflow_dataset(
            df, batch_size=cfg["data"]["batch_size"], image_size=(size, size), shuffle=False
        )
        for name, df in zip(["train", "val", "test"], splits)
    }
    class_names = {lvl: sorted(df_train[lvl].unique()) for lvl in LEVELS}
    n_classes = tuple(len(class_names[lvl]) for lvl in LEVELS)
    parent_genus, parent_species = make_parent_lists(
        df_train["family"].tolist(), df_train["genus"].tolist(), df_train["species"].tolist()
    )

    out = out_root / f"{arch}_{datetime.datetime.now().strftime('%y%m.%d.%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {arch.upper()} | {out} =====")
    run_cfg = {**cfg, "model": {**cfg["model"], "architecture": arch}}
    (out / "config.yml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2))

    tf.keras.backend.clear_session()
    tf.random.set_seed(cfg["general"]["seed"])
    model = build_model(tf, cfg, arch, n_classes, parent_genus, parent_species)

    # Stage 1: backbone frozen
    stage = cfg["training"]
    compile_model(tf, cfg, model, stage)
    print(f"\nStage 1/2: backbone frozen, up to {stage['epochs']} epochs")
    hist = model.fit(
        tf_ds["train"], validation_data=tf_ds["val"], epochs=stage["epochs"],
        callbacks=callbacks(tf, stage, out / "history.csv"), verbose=0,
    )
    (out / "history.json").write_text(json.dumps(hist.history, default=float))
    model.save_weights(str(out / "final.weights.h5"))
    stage1_eval, _ = evaluate(model, tf_ds["test"], df_test, class_names, parent_genus, parent_species)
    (out / "evaluations.json").write_text(json.dumps(stage1_eval, indent=2))
    print("Stage 1 test:", _overall(stage1_eval["overall"]))

    # Stage 2: unfreeze the top backbone layers
    stage = cfg["finetune"]
    unfreeze_top_layers(model, stage["unfreeze_last_layers"])
    compile_model(tf, cfg, model, stage)
    print(
        f"\nStage 2/2: fine-tuning the top {stage['unfreeze_last_layers']} backbone layers, "
        f"up to {stage['epochs']} epochs"
    )
    hist = model.fit(
        tf_ds["train"], validation_data=tf_ds["val"], epochs=stage["epochs"],
        callbacks=callbacks(tf, stage, out / "finetune_history.csv"), verbose=0,
    )
    (out / "finetune_history.json").write_text(json.dumps(hist.history, default=float))
    model.save_weights(str(out / "finetuned.weights.h5"))
    final_eval, preds = evaluate(model, tf_ds["test"], df_test, class_names, parent_genus, parent_species)
    (out / "finetuned_evaluations.json").write_text(json.dumps(final_eval, indent=2))
    preds.to_csv(out / "finetuned_predictions.csv", index=False)
    plot_history(out)
    print("Final test:  ", _overall(final_eval["overall"]))
    return final_eval["overall"]


def _overall(metrics):
    return "  ".join(f"{k} {v:.4f}" for k, v in metrics.items())


def plot_history(out):
    """Save loss and per-level accuracy over both stages as training_curves.png."""
    from matplotlib.figure import Figure  # no pyplot: works without a display

    history = pd.concat(
        [pd.read_csv(out / "history.csv"), pd.read_csv(out / "finetune_history.csv")], ignore_index=True
    )
    stage1_epochs = len(pd.read_csv(out / "history.csv"))
    panels = ["loss"] + [f"{lvl}_accuracy" for lvl in LEVELS]
    fig = Figure(figsize=(4 * len(panels), 3.5))
    for ax, metric in zip(fig.subplots(1, len(panels)), panels):
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history[metric], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.axvline(stage1_epochs + 0.5, color="gray", ls=":", lw=0.8)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("epoch (dotted line: start of fine-tuning)")
    fig.axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)


def main():
    args = parse_args()
    cfg = load_config(args)
    out_root = Path(cfg["general"]["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    start_log(out_root)
    tf = setup_tf(cfg["general"])
    splits = load_splits(cfg, args.max_per_species)

    arch = cfg["model"]["architecture"]
    summary_path = out_root / "summary.csv"
    for a in ARCHS if arch == "all" else [arch]:
        overall = run(tf, cfg, a, splits, out_root)
        pd.DataFrame([{"architecture": a.upper(), **overall}]).to_csv(
            summary_path, mode="a", header=not summary_path.exists(), index=False
        )
    print(f"\nSummary of runs in {out_root}:")
    print(pd.read_csv(summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
