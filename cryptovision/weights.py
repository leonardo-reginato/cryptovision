"""
Build CryptoVision architectures and load the released trained weights.

    from cryptovision.weights import load_pretrained, class_names
    model = load_pretrained("gated")     # std | concat | gated | att
    names = class_names()                # {"family": [...], "genus": [...], "species": [...]}
    fam, gen, spe = model.predict(images)  # images: float32, 0-255, shape (N, 352, 352, 3)

    model, names = load_trained("outputs/gated_2611.02.1015")   # a model you trained

models/manifest.json lists each architecture's weight file, download URL and SHA-256;
models/class_names.json maps output indices to taxon names.
"""

import hashlib
import json
from pathlib import Path
import urllib.request

import yaml

from cryptovision.models import CryptoVisionModels

PACKAGE_DIR = Path(__file__).parent
SETTINGS = PACKAGE_DIR / "settings.yml"
# shortcut: models/ sits next to the package, so this assumes a source checkout
# (pip install -e .); move the metadata into the package if wheels are published
MODELS_DIR = PACKAGE_DIR.parent / "models"
LEVELS = ("family", "genus", "species")


def build_model(model_cfg, architecture, n_classes, augmentation=None, backbone_weights="imagenet"):
    """CryptoVision architecture from the `model` section of settings.yml."""
    return CryptoVisionModels.basic(
        imagenet_name=model_cfg["backbone"],
        augmentation=augmentation,
        input_shape=(model_cfg["image_size"], model_cfg["image_size"], 3),
        shared_dropout=model_cfg["shared_dropout"],
        feat_dropout=model_cfg["features_dropout"],
        shared_layer_neurons=model_cfg["shared_layer_neurons"],
        se_block=model_cfg["use_se_block"],
        pooling_type=model_cfg["pooling_type"],
        architecture=architecture,
        output_neurons=n_classes,
        backbone_weights=backbone_weights,
    )


def unfreeze_top_layers(model, n_layers):
    """Make the last n_layers of the backbone trainable and freeze the rest (stage 2)."""
    import tensorflow as tf

    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential)
    )
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False


def class_names(models_dir=MODELS_DIR, model="gated"):
    """Taxon names in output order for a released model (see its manifest entry)."""
    models_dir = Path(models_dir)
    entry = json.loads((models_dir / "manifest.json").read_text())["models"].get(model, {})
    return json.loads((models_dir / entry.get("class_names", "class_names.json")).read_text())


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pretrained(name, models_dir=MODELS_DIR, settings=SETTINGS):
    """
    Build a released model and load its weights. `name` is a key of models/manifest.json:
    std | concat | gated | att (Table 1), or web_only | mixed (Fig 2 data comparison).
    """
    models_dir = Path(models_dir)
    entry = json.loads((models_dir / "manifest.json").read_text())["models"][name]
    path = models_dir / entry["file"]
    if not path.exists():
        if not entry.get("url"):
            raise FileNotFoundError(f"{path} is missing and manifest.json has no download URL for it")
        print(f"Downloading {entry['url']} -> {path}")
        urllib.request.urlretrieve(entry["url"], path)
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}: delete the file and download it again")

    cfg = yaml.safe_load(Path(settings).read_text())
    cfg["model"]["image_size"] = entry.get("image_size", cfg["model"]["image_size"])
    return _build_and_load(cfg, entry.get("architecture", name), class_names(models_dir, name), path)


def load_trained(run_dir, weights="finetuned.weights.h5"):
    """
    Load a model you trained with cryptovision.train, from its output folder.
    Returns (model, class_names) for use with predict.CryptoVisionAI.
    """
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yml").read_text())
    names = json.loads((run_dir / "class_names.json").read_text())
    return _build_and_load(cfg, cfg["model"]["architecture"], names, run_dir / weights), names


def _build_and_load(cfg, architecture, names, path):
    model = build_model(
        cfg["model"], architecture, tuple(len(names[k]) for k in LEVELS), backbone_weights=None
    )
    # Weight files are saved after fine-tuning, and .h5 files store each layer's trainable
    # weights before its frozen ones; mirror that split so the weights load in order
    unfreeze_top_layers(model, cfg["finetune"]["unfreeze_last_layers"])
    model.load_weights(str(path))
    model.trainable = False
    return model
