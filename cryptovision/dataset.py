from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from cryptovision.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

src_path = '/Volumes/T7_shield/CryptoVision/Data/Images/Sources'

catalog = {
    'sjb': {
        'path': f'{src_path}/Lab/SJB/Processed/Species/v241226/uniques_catalog.csv',
    },
    'scls': {
        'path': f'{src_path}/Lab/SCLS/Processed/Species/v250115/images_catalog.csv',
    },
    'lirs': {
        'path': f'{src_path}/Lab/LIRS23/Processed/Species/v250115/images_catalog.csv',
    },
    'cbc': {
        'path': f'{src_path}/Lab/CBC24/Processed/Species/v250115/images_catalog.csv',
    },
    'web': {
        'path': f'{src_path}/Web/Species/v240712/uniques_catalog.csv',
    },
    'inat': {
        'path': f'{src_path}/INaturaList/Species/v250128/images_catalog_v250211.csv',
    }
}


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    min_samples: int = 10,
    # ----------------------------------------------
):
    datasets = []
    
    for name, data in catalog.items():
        logger.info(f"Processing {name} dataset")
        datasets.append(pd.read_csv(data['path']))
        
    data = pd.concat(datasets, axis=0, ignore_index=True)
    
    logger.info(f"Dataset shape before: {data.shape}")
    logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")
    
    data = data.drop_duplicates(subset='hash', keep='first')
    logger.info(f"Number of duplicates: {data.shape[0] - data.drop_duplicates(subset='hash', keep='first').shape[0]}")
    
    data = data[data['species'].map(data['species'].value_counts()) > min_samples]
    logger.info(f"Filtration by species with more than {min_samples} images")
    logger.info(f"Dataset shape after: {data.shape}")
    logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")
    
    data = data[~data['flag_small']]
    logger.info(f"Filtration by small images {data.shape[0] - data[~data['flag_small']].shape[0]}")
    
    return data

if __name__ == "__main__":
    app()
