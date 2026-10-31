# CryptoVision

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23827343.svg)](https://doi.org/10.5281/zenodo.23827343)

Taxonomy-aware deep learning for identifying cryptic coral reef fishes at three levels
(family, genus, species) from images.

Code for: Reginato et al. (2026), *Integrating deep learning, biological hierarchies, and
high-resolution imagery to create a new identification tool for cryptic coral reef fishes*.
PLOS ONE. https://doi.org/10.1371/journal.pone.0349646

## What is here

```
cryptovision/
  data.py           image catalogs, quality control, 70/15/15 split, tf.data pipelines
  models.py         ResNet50V2 + SE block with the four fusion designs: std, concat, gated, att
  losses.py         Taxonomy-Focal Cross-Loss (TFCL), Eqs. 1-5 of the paper
  train.py          two-stage training + test evaluation (Table 1 metrics and TAS)
  weights.py        load the released models, or models you trained
  predict.py        prediction and saliency maps
  settings.yml      every setting used for the paper's experiments
environments/       pinned conda environments (Linux/WSL2, macOS, Windows)
models/             released weights (*.h5), manifest.json and class names
images/             three example lab photos (Eviota albolineata, E. infulata, E. teresae)
notebooks/
  01_quickstart.ipynb        identify fish with a released model, saliency maps
  02_evaluate_models.ipynb   evaluate the released models: web-only vs mixed, the four
                             architectures, McNemar tests, error vs genus richness, saliency
```

Commands and paths below are relative to the repository root.

## Install

The released weights in `models/` are stored with [Git LFS](https://git-lfs.com). Install it
before cloning, or the `.h5` files arrive as small pointer files:

```bash
git lfs install
git clone https://github.com/leonardo-reginato/cryptovision.git
cd cryptovision
```

If you cloned without Git LFS, run `git lfs pull` inside the repository. Each weight file is
about 110 MB, over GitHub's 100 MB limit for regular files, so Git LFS is also required to
commit or push changes to `models/*.h5` (tracked in `.gitattributes`).

Create the environment that matches your machine, then install the package:

```bash
conda env create -f environments/linux.yml     # Linux or WSL2, NVIDIA GPU (TF 2.16)
conda env create -f environments/macos.yml     # macOS, Apple Silicon (TF 2.16 + Metal)
conda env create -f environments/windows.yml   # native Windows, NVIDIA GPU (TF 2.10)
conda activate cryptovision
pip install -e .
```

Check the GPU is visible:

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Native Windows is limited to TensorFlow 2.10 (Keras 2), the last release with native
Windows GPU support; the code handles the differences automatically. Linux/WSL2 and macOS
use TensorFlow 2.16 (Keras 3), the stack used for the paper's models.

## Identify fish with a released model

```python
from cryptovision.predict import CryptoVisionAI
from cryptovision.weights import class_names, load_pretrained

names = class_names(model="gated")
ai = CryptoVisionAI(load_pretrained("gated"), names["family"], names["genus"], names["species"])
ai.predict("images/lab_Eviota_albolineata_sample.jpg")    # (family, genus, species)
```

Released models (`models/manifest.json`):

| name | trained on | species |
|---|---|---|
| `std`, `concat`, `gated`, `att` | the paper's full dataset, one per fusion design | 113 |
| `web_only`, `mixed` | size-matched web-only and 50:50 web/lab sets | 106 |

`load_pretrained` checks each file's SHA-256 and downloads it from the manifest's URL if it
is missing. See `notebooks/01_quickstart.ipynb` for confidences and saliency maps.

## Evaluate the released models

`notebooks/02_evaluate_models.ipynb` evaluates the released models on their test splits:
precision, recall, accuracy, F1 and TAS for web-only vs mixed training data and for the four
architectures, paired McNemar tests, species-level error vs genus richness, and saliency maps
for the photos in `images/`. It needs the two catalogs of the paper's data and the images they
list; set their locations at the top of the notebook:

```python
CATALOG = "path/to/paper_catalog.csv"
MIX_CATALOG = "path/to/mixcompare_catalog.csv"
```

## Train your own model

**1. Organise your images** in one folder, one subfolder per family, genus and species:

```
my_images/
  Gobiidae/
    Eviota/
      Eviota_afelei/     img001.jpg, img002.jpg, ...
      Eviota_zebrina/    ...
  Blenniidae/
    Ecsenius/
      Ecsenius_bicolor/  ...
```

**2. Train** (both stages, then evaluation on the test split):

```bash
python -m cryptovision.train --catalog my_images/ --arch gated     # std | concat | gated | att | all
```

The first run writes `my_images/catalog.csv` (one row per image; see *Catalog format*) and
prints the number of images per species. Duplicate images are removed and species with 90 or
fewer images are dropped. For a smaller dataset, copy `cryptovision/settings.yml`, lower
`data.min_samples_per_class` and add `--config my_settings.yml`; every other setting (epochs,
learning rates, image size, loss weights) is in the same file. Without `--catalog`, the data
location is read from `data.catalog`, which defaults to the `CRYPTOVISION_DATA` environment
variable.

**3. Check the results** in `outputs/<arch>_<timestamp>/`:

| file | content |
|---|---|
| `finetuned_evaluations.json` | test metrics per level and overall, including TAS |
| `finetuned_predictions.csv` | prediction and confidence for every test image |
| `training_curves.png` | loss and accuracy per epoch, both stages |
| `finetuned.weights.h5` | trained weights (`final.weights.h5`: after stage 1 only) |
| `config.yml`, `class_names.json` | the settings and class order of this run |

Console output is saved to `outputs/logs/`.

**4. Use your model** like a released one:

```python
from cryptovision.predict import CryptoVisionAI
from cryptovision.weights import load_trained

model, names = load_trained("outputs/gated_<timestamp>")   # the run folder from step 3
ai = CryptoVisionAI(model, names["family"], names["genus"], names["species"])
```

and continue as in `notebooks/01_quickstart.ipynb`.

Hardware: training caches decoded images in memory (about 30 GB for the paper's 18,559
training images at 352 px); one architecture takes about 1.5 h on a recent NVIDIA GPU.

### Catalog format

Instead of an image folder you can pass a catalog CSV (`--catalog my_data/catalog.csv`), or
list several catalogs and folders under `data.catalog` in the settings file to combine them.

| column | required | content |
|---|---|---|
| `image_path` | yes | image path, relative to the catalog's folder (or absolute) |
| `family`, `genus`, `species` | yes | taxon names (`_` in species folder names becomes a space) |
| `hash` | no | image content hash, used to drop duplicates (written by `make_catalog`) |
| `split` | no | `train` / `val` / `test` to fix the split; rows are used in catalog order |
| `source` | no | free text, e.g. the dataset each image came from |

Without a `split` column the data is split 70/15/15, stratified by species (seed 42). After
adding images to a folder, rebuild its catalog with
`cryptovision.data.make_catalog("my_images/")`.

## The paper's data

The paper's dataset is described by `paper_catalog.csv`: 26,514 images of 113 species,
62 genera and 20 families, with the exact train / validation / test split.
`mixcompare_catalog.csv` describes the web-only vs mixed comparison set (10,600 images,
106 species).

- **Laboratory images**: the processed lab-image dataset is on Figshare,
  https://doi.org/10.6084/m9.figshare.32305065, and loads with `data.processed_npz_dataset`.
- **Web images** (FishBase, iNaturalist) belong to their contributors and cannot be
  redistributed. They remain available from those platforms under their own licenses.
- Images were curated by hand (selection, cropping, label checks), so the exact training
  set cannot be rebuilt from public sources. The released weights make the reported
  evaluation reproducible without retraining (`notebooks/02_evaluate_models.ipynb`).

Training follows the paper's protocol: stage 1 with the ResNet50V2 frozen (15 epochs, Adam 5e-4,
LR x0.2 at epochs 6 and 11), stage 2 with the top 75 backbone layers unfrozen (10 epochs,
RMSprop 1e-5), then balanced accuracy and weighted precision / recall / F1 on the joint
family-genus-species label, plus the Taxonomic Alignment Score (TAS).

## Loss: Taxonomy-Focal Cross-Loss

`cryptovision/losses.py` implements Eqs. 1-5 of the paper. Each head gets a focal
cross-entropy term on the probability of its true class; the genus and species losses add a
penalty on the parent head's probability for the true parent taxon:

```
L_fam  = -α (1 - p_fam(y_fam))^γ · log p_fam(y_fam)
L_gen  = -α (1 - p_gen(y_gen))^γ · log p_gen(y_gen) + λ_gen · [ -log p_fam(parent(y_gen)) ]
L_spe  = -α (1 - p_spe(y_spe))^γ · log p_spe(y_spe) + λ_spe · [ -log p_gen(parent(y_spe)) ]
L_TFCL = L_fam + L_gen + L_spe
```

Values in `cryptovision/settings.yml`: α = 0.1, γ = 2, λ_gen = λ_spe = 0.1.

The released weights in `models/` were trained with an earlier version of this loss that also
applied label smoothing (0.1) to the focal terms; models trained with this repository use the
loss exactly as written above.

## Citation

```bibtex
@article{reginato2026cryptovision,
  title   = {Integrating deep learning, biological hierarchies, and high-resolution imagery
             to create a new identification tool for cryptic coral reef fishes},
  author  = {Reginato, Leonardo and others},
  journal = {PLOS ONE},
  year    = {2026},
  doi     = {10.1371/journal.pone.0349646}
}
```

To cite the code and released models (archived on Zenodo):

```bibtex
@software{reginato2026cryptovision_code,
  title     = {CryptoVision},
  author    = {Reginato, Leonardo},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23827343}
}
```

## License

MIT, see `LICENSE`.
