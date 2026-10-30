"""
Datasets for CryptoVision.

Images are organised in one folder, one level per taxonomic rank:

    my_images/<family>/<genus>/<species>/<image files>
    e.g. my_images/Gobiidae/Eviota/Eviota_afelei/img001.jpg

make_catalog() scans that folder and writes a catalog CSV (one row per image) that
training reads. A catalog can also be written by hand; it needs the columns
image_path, family, genus and species, and may add:
    hash    content hash, used to drop duplicate images
    split   train / val / test, to fix the split (rows are used in catalog order)
    source  free text, e.g. the dataset each image came from
Relative image paths are resolved against the catalog's folder.
"""

import hashlib
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

LEVELS = ["family", "genus", "species"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CATALOG_NAME = "catalog.csv"


def make_catalog(image_dir, catalog_path=None) -> pd.DataFrame:
    """
    Scan image_dir/<family>/<genus>/<species>/<image> and write a catalog CSV
    (default: image_dir/catalog.csv). Species folder names may use "_" for spaces.
    """
    image_dir = Path(image_dir)
    catalog_path = Path(catalog_path) if catalog_path else image_dir / CATALOG_NAME
    rows = []
    for path in sorted(image_dir.glob("*/*/*/*")):
        parts = path.relative_to(image_dir).parts
        if path.suffix.lower() not in IMAGE_EXTENSIONS or any(p.startswith(".") for p in parts):
            continue
        family, genus, species = parts[:3]
        rows.append(
            {
                "image_path": Path(os.path.relpath(path, catalog_path.parent)).as_posix(),
                "family": family,
                "genus": genus,
                "species": species.replace("_", " "),
                "hash": hashlib.md5(path.read_bytes()).hexdigest(),
            }
        )
    if not rows:
        raise ValueError(
            f"No images found in {image_dir}. Expected <family>/<genus>/<species>/<image> "
            f"with extensions {sorted(IMAGE_EXTENSIONS)}."
        )
    catalog = pd.DataFrame(rows)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    print(
        f"Wrote {catalog_path}: {len(catalog)} images, {catalog['family'].nunique()} families, "
        f"{catalog['genus'].nunique()} genera, {catalog['species'].nunique()} species."
    )
    return catalog


def load_catalogs(catalogs) -> pd.DataFrame:
    """
    Read one or more catalogs (CSV paths, or image folders whose catalog.csv is
    built on first use) and return one table with absolute image paths.
    """
    frames = []
    for item in [catalogs] if isinstance(catalogs, (str, Path)) else catalogs:
        path = Path(item)
        if path.is_dir():
            path = path / CATALOG_NAME
            if not path.exists():
                make_catalog(path.parent, path)
        df = pd.read_csv(path)
        missing = [c for c in ["image_path"] + LEVELS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        df["image_path"] = [str(path.parent / p) for p in df["image_path"]]  # absolute paths stay as-is
        frames.append(df)
        print(f"Loaded {path} with {len(df)} images.")
    return pd.concat(frames, ignore_index=True)


def quality_control(
    data: pd.DataFrame, min_samples_per_class: int, remove_duplicates: bool = True,
    stratify_by: str = "species",
) -> pd.DataFrame:
    """Drop duplicate images (by hash) and classes with min_samples_per_class images or fewer."""
    if remove_duplicates and "hash" in data.columns:
        n = len(data)
        data = data.drop_duplicates(subset="hash", keep="first")
        print(f"Removed {n - len(data)} duplicate images.")
    counts = data[stratify_by].value_counts()
    keep = counts[counts > min_samples_per_class].index
    n = len(data)
    data = data[data[stratify_by].isin(keep)]
    print(
        f"Removed {n - len(data)} images of {len(counts) - len(keep)} {stratify_by} classes "
        f"with {min_samples_per_class} images or fewer; {len(data)} images remain."
    )
    return data.reset_index(drop=True)


def split_data(
    data: pd.DataFrame, test_fraction: float = 0.15, val_fraction: float = 0.15,
    seed: int = 42, stratify_by: str = "species",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train / validation / test split. Uses the catalog's 'split' column when present;
    otherwise a stratified split with the given fractions of the full data.
    """
    if "split" in data.columns:
        train, val, test = (data[data["split"] == s] for s in ("train", "val", "test"))
    else:
        train_val, test = train_test_split(
            data, test_size=test_fraction, stratify=data[stratify_by], random_state=seed
        )
        train, val = train_test_split(
            train_val,
            test_size=val_fraction / (1 - test_fraction),
            stratify=train_val[stratify_by],
            random_state=seed,
        )
    print(f"Split: train {len(train)}, validation {len(val)}, test {len(test)} images.")
    return train, val, test


def load_image(image_path, image_size: tuple[int, int] = None):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    if image_size:
        img = tf.image.resize(img, image_size)

    return img


def process_image_and_labels(
    image_path,
    family,
    genus,
    species,
    family_labels,
    genus_labels,
    species_labels,
    image_size=(224, 224),
):
    """
    Load one image and one-hot encode its family, genus and species labels.

    The *_labels tensors hold the sorted class names of each level.
    Returns (image, {"family": ..., "genus": ..., "species": ...}).
    """
    img = load_image(image_path, image_size)

    family_label = tf.argmax(tf.equal(family_labels, family))
    genus_label = tf.argmax(tf.equal(genus_labels, genus))
    species_label = tf.argmax(tf.equal(species_labels, species))

    family_label = tf.one_hot(family_label, len(family_labels))
    genus_label = tf.one_hot(genus_label, len(genus_labels))
    species_label = tf.one_hot(species_label, len(species_labels))

    return img, {"family": family_label, "genus": genus_label, "species": species_label}


def tensorflow_dataset(
    df,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build a batched, cached tf.data.Dataset of (image, labels) from a DataFrame
    with 'image_path', 'family', 'genus' and 'species' columns.

    Class indices follow the sorted unique names of each level in df.
    """
    family_labels = tf.constant(sorted(df["family"].unique()))
    genus_labels = tf.constant(sorted(df["genus"].unique()))
    species_labels = tf.constant(sorted(df["species"].unique()))

    path_ds = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path"].values,
            df["family"].values,
            df["genus"].values,
            df["species"].values,
        )
    )

    image_dataset = path_ds.map(
        lambda path, family, genus, species: process_image_and_labels(
            path,
            family,
            genus,
            species,
            family_labels,
            genus_labels,
            species_labels,
            image_size=image_size,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    image_dataset = (
        image_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    if shuffle:
        image_dataset = image_dataset.shuffle(buffer_size=len(df))

    return image_dataset


def processed_npz_dataset(
    npz_path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Build a TensorFlow dataset from a Figshare processed NPZ split
    (e.g. data/train.npz), yielding images and one-hot taxonomic labels.
    """
    data = np.load(npz_path, allow_pickle=True)
    images = data["images"].astype(np.float32)
    labels = {
        "family": tf.one_hot(
            data["family"].astype(np.int64), int(data["n_family"].item())
        ),
        "genus": tf.one_hot(
            data["genus"].astype(np.int64), int(data["n_genus"].item())
        ),
        "species": tf.one_hot(
            data["species"].astype(np.int64), int(data["n_species"].item())
        ),
    }

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), seed=seed)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


def make_parent_lists(family_labels, genus_labels, species_labels):
    """
    From parallel per-sample label lists, return
      parent_genus:   list of length G, genus_idx -> family_idx
      parent_species: list of length S, species_idx -> genus_idx
    using the same sorted class order as tensorflow_dataset.
    """
    families = sorted(set(family_labels))
    genera = sorted(set(genus_labels))
    species = sorted(set(species_labels))

    fam2idx = {fam: i for i, fam in enumerate(families)}
    gen2idx = {gen: i for i, gen in enumerate(genera)}

    genus2family = {gen: fam for gen, fam in set(zip(genus_labels, family_labels))}
    species2genus = {spe: gen for spe, gen in set(zip(species_labels, genus_labels))}

    parent_genus = [fam2idx[genus2family[gen]] for gen in genera]
    parent_species = [gen2idx[species2genus[spe]] for spe in species]

    return parent_genus, parent_species
