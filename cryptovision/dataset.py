from pathlib import Path

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
        'path': f'{src_path}/INaturaList/Species/v250116/images_catalog.csv',
        'clean_path': f'{src_path}/INaturaList/Species/v250128/images_catalog.csv',
    }
}


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
