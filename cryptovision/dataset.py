import os
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


class DatasetLoader:
    def __init__(
        self,
        sources: Dict[str, str],
        min_samples: int = 10,
        stratify_col: str = "species",
        verbose: bool = True,
    ):
        """
        Initialize the DatasetLoader.

        :param sources: Dictionary mapping dataset names to directory paths containing a catalog.csv file.
        :param min_samples: Minimum number of samples required for a category to be retained.
        :param stratify_col: Column name to use for stratified splitting.
        :param verbose: If True, log progress details.
        """
        self.sources = sources
        self.min_samples = min_samples
        self.stratify_col = stratify_col
        self.verbose = verbose

    def load_catalog(self, dir_path: str) -> pd.DataFrame:
        """Load the catalog CSV from a given directory."""
        catalog_path = Path(dir_path) / "catalog.csv"
        if not catalog_path.exists():
            logger.error(f"Catalog file not found: {catalog_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(catalog_path)
            if self.verbose:
                logger.info(
                    f"Loaded catalog from {catalog_path} with {df.shape[0]} records."
                )
            return df
        except Exception as e:
            logger.error(f"Error reading catalog file {catalog_path}: {str(e)}")
            return pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """Load and combine data from all provided sources."""
        datasets = []
        for name, path in self.sources.items():
            df = self.load_catalog(path)
            if not df.empty:
                df["source"] = name  # Tag each record with its source
                datasets.append(df)
        if not datasets:
            raise ValueError("No data loaded from any source!")
        data = pd.concat(datasets, ignore_index=True)
        if self.verbose:
            logger.info(f"Combined dataset size: {data.shape[0]}")
        return data

    def apply_quality_control(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply duplicate removal and filtering based on sample count and quality flags."""
        # Remove duplicate entries based on 'hash' if available
        if "hash" in data.columns:
            initial_count = data.shape[0]
            data = data.drop_duplicates(subset="hash", keep="first")
            if self.verbose:
                logger.info(
                    f"Removed {initial_count - data.shape[0]} duplicates based on 'hash'."
                )

        # Filter out categories with fewer than min_samples entries
        if self.stratify_col in data.columns:
            counts = data[self.stratify_col].value_counts()
            valid_categories = counts[counts > self.min_samples].index
            filtered = data[data[self.stratify_col].isin(valid_categories)]
            if self.verbose:
                removed = data.shape[0] - filtered.shape[0]
                logger.info(
                    f"Filtered out {removed} records with insufficient samples in '{self.stratify_col}'."
                )
            data = filtered

        # Exclude images flagged as small if the flag exists
        if "flag_small" in data.columns:
            initial_count = data.shape[0]
            data = data[~data["flag_small"]]
            if self.verbose:
                logger.info(
                    f"Removed {initial_count - data.shape[0]} records flagged as small images."
                )

        if self.verbose:
            logger.info(f"Dataset size after quality control: {data.shape[0]}")
        return data

    def split_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training, validation, and test sets using stratification.

        :param data: DataFrame to split.
        :param test_size: Proportion of data to allocate to the test set.
        :param val_size: Proportion of the remaining training data to allocate to validation.
        :param random_state: Random seed for reproducibility.
        :return: Tuple of DataFrames (train, validation, test).
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < val_size < 1:
            raise ValueError("val_size must be between 0 and 1")

        stratify_series = (
            data[self.stratify_col] if self.stratify_col in data.columns else None
        )
        train_val, test = train_test_split(
            data,
            test_size=test_size,
            stratify=stratify_series,
            random_state=random_state,
        )

        # Adjust validation size relative to the remaining data after removing the test set.
        val_ratio = val_size / (1 - test_size)
        stratify_series = (
            train_val[self.stratify_col]
            if self.stratify_col in train_val.columns
            else None
        )
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=stratify_series,
            random_state=random_state,
        )

        if self.verbose:
            logger.info(
                f"Split data into train ({train.shape[0]}), validation ({val.shape[0]}), and test ({test.shape[0]}) records."
            )
        return train, val, test


def check_image_path(df: pd.DataFrame, src_path: str) -> pd.DataFrame:
    df['image_path'] = df['image_path'].apply(
        lambda x: x.replace(
            '/Volumes/T7_shield/CryptoVision/Data/Sources',
            src_path,
        )
    )
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataset(
    src_path: str,
    min_samples: int = 10,
    verbose: bool = True,
    return_split: bool = False,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify_by: str = "species",
    random_state: int = 42,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Load dataset by reading from multiple sources, apply quality control,
    and optionally split into train/validation/test sets.

    :param src_path: Base directory path for the data sources.
    :param min_samples: Minimum sample count for a category to be included.
    :param verbose: If True, log progress information.
    :param return_split: If True, return a tuple (train, validation, test) instead of the full dataset.
    :param test_size: Proportion for the test split.
    :param val_size: Proportion for the validation split.
    :param stratify_by: Column name to use for stratification.
    :param random_state: Seed for reproducibility.
    :return: A DataFrame or a tuple of DataFrames (train, validation, test).
    """
    if not os.path.exists(src_path):
        raise ValueError(f"Source path does not exist: {src_path}")

    # Define the default catalog sources using a consistent directory structure.
    default_sources = {
        "sjb": os.path.join(src_path, "SJB/Species/v241226"),
        "scls": os.path.join(src_path, "SCLS/Species/v250115"),
        "lirs": os.path.join(src_path, "LIRS23/Species/v250115"),
        "cbc": os.path.join(src_path, "CBC24/Species/v250115"),
        "web": os.path.join(src_path, "WEB/Species/v240712"),
        "inat": os.path.join(src_path, "INaturaList/Species/v250128"),
    }

    loader = DatasetLoader(
        sources=default_sources,
        min_samples=min_samples,
        stratify_col=stratify_by,
        verbose=verbose,
    )
    data = loader.load_data()
    data = loader.apply_quality_control(data)
    data = check_image_path(data, src_path)
    if return_split:
        return loader.split_data(
            data, test_size=test_size, val_size=val_size, random_state=random_state
        )
    else:
        return data


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def cli(
        src_path: str = typer.Argument(
            ..., help="Base directory path for the data sources"
        ),
        min_samples: int = typer.Option(
            10, help="Minimum sample count for a category to be included"
        ),
        return_split: bool = typer.Option(
            False, help="Return train/validation/test splits instead of full dataset"
        ),
        test_size: float = typer.Option(0.2, help="Proportion for the test split"),
        val_size: float = typer.Option(0.2, help="Proportion for the validation split"),
        stratify_by: str = typer.Option(
            "species", help="Column name to use for stratification"
        ),
        random_state: int = typer.Option(42, help="Random seed for reproducibility"),
    ) -> None:
        """
        CLI interface for loading and processing the dataset.
        """
        result = load_dataset(
            src_path,
            min_samples,
            True,
            return_split,
            test_size,
            val_size,
            stratify_by,
            random_state,
        )
        if return_split:
            train, val, test = result
            logger.info(
                f"Train size: {train.shape}, Validation size: {val.shape}, Test size: {test.shape}"
            )
        else:
            logger.info(f"Loaded dataset size: {result.shape}")

    app()
