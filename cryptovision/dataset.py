from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
import os

from cryptovision.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from cryptovision import tools

app = typer.Typer()

src_path = '/Volumes/SANDISK/CryptoVision/Sources'

catalog = {
    'sjb': {
        'dir': f'{src_path}/SJB/Species/v241226/',
    },
    'scls': {
        'dir': f'{src_path}/SCLS/Species/v250115/',
    },
    'lirs': {
        'dir': f'{src_path}/LIRS23/Species/v250115/',
    },
    'cbc': {
        'dir': f'{src_path}/CBC24/Species/v250115/',
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
    return_split: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify_by: str = 'species',
    random_state: int = 42,
):
    datasets = []
    
    for name, data in catalog.items():
        
        dataset = pd.read_csv(os.path.join(data['dir'], 'images_catalog.csv'))
        
        if verbose:
            logger.info(f"DataSet {name} loaded -> Size: {dataset.shape}")
            
        datasets.append(dataset)
        
    data = pd.concat(datasets, axis=0, ignore_index=True)
    
    og_rows = data.shape[0]
    
    if verbose:
        logger.info(f"Initial DataSet Size: {og_rows:,}")
        logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")
        
    data = data.drop_duplicates(subset='hash', keep='first')
    
    if verbose:
        logger.info(f"Number of duplicates: {og_rows - data.shape[0]:,}")
    
    data = data[data['species'].map(data['species'].value_counts()) > min_samples]
    
    if verbose:
        logger.info(f"Filtration by species with more than {min_samples} images")
        
        
    data = data[~data['flag_small']]
    
    if verbose:
        logger.info(f"Filtration by small images {og_rows - data[~data['flag_small']].shape[0]}")
        logger.info(f"Dataset shape after: {data.shape[0]:,}")
        logger.info(f"Fam {data['family'].nunique()} | Gen {data['genus'].nunique()} | Spec {data['species'].nunique()}")

    if return_split:
        
        train, val, test = tools.split_dataframe(
            data,
            test_size=test_size,
            val_size=val_size,
            stratify_by=stratify_by,
            random_state=random_state
        )
        return train, val, test
    
    else:
        return data

if __name__ == "__main__":
    app()
    