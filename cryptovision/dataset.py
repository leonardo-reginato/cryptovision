from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
import os

from cryptovision.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

src_path = '/Volumes/T7_shield/CryptoVision/Data/Sources'

catalog = {
    'sjb': {
        'dir': f'{src_path}/Lab/SJB/Processed/Species/v241226/',
    },
    'scls': {
        'dir': f'{src_path}/Lab/SCLS/Processed/Species/v250115/',
    },
    'lirs': {
        'dir': f'{src_path}/Lab/LIRS23/Processed/Species/v250115/',
    },
    'cbc': {
        'dir': f'{src_path}/Lab/CBC24/Processed/Species/v250115/',
    },
    'web': {
        'dir': f'{src_path}/Web/Species/v240712/',
    },
    'inat': {
        'dir': f'{src_path}/INaturaList/Species/v250128/',
    }
}


@app.command()
def main(
    min_samples: int = 10,
    verbose: bool = True,
):
    datasets = []
    
    for name, data in catalog.items():
        if verbose:
            dataset = pd.read_csv(os.path.join(data['dir'], 'images_catalog.csv'))
            logger.info(f"DataSet {name} loaded -> Size: {dataset.shape}")
        datasets.append(dataset)
        
    data = pd.concat(datasets, axis=0, ignore_index=True)
    
    if verbose:
        logger.info(f"Initial DataSet Size: {data.shape[0]:,}")
        logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")
        
    data = data.drop_duplicates(subset='hash', keep='first')
    if verbose:
        logger.info(f"Number of duplicates: {data.shape[0] - data.drop_duplicates(subset='hash', keep='first').shape[0]}")
    
    data = data[data['species'].map(data['species'].value_counts()) > min_samples]
    
    if verbose:
        logger.info(f"Filtration by species with more than {min_samples} images")
        
        
    data = data[~data['flag_small']]
    
    if verbose:
        logger.info(f"Filtration by small images {data.shape[0] - data[~data['flag_small']].shape[0]}")
        logger.info(f"Dataset shape after: {data.shape[0]:,}")
        logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")
    
    
    if verbose:
        logger.info(f"Filtration by small images {data.shape[0] - data[~data['flag_small']].shape[0]}")
    


    return data

if __name__ == "__main__":
    app()