import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

# Configure logger
logger.add("logs/web_scraping.log", rotation="500 MB")


class INaturalistScraper:
    """A class to handle iNaturalist API interactions and image downloads."""

    BASE_URL = "https://api.inaturalist.org/v1/observations"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.5
    RATE_LIMIT_DELAY = 1  # seconds between requests

    def __init__(self):
        """Initialize the scraper with a configured session."""
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry mechanism."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_observations(
        self, taxon_name: str, rank: str, per_page: int = 30, page: int = 1
    ) -> Optional[Dict]:
        """
        Fetch observations for a given taxon name and rank from iNaturalist API.

        Args:
            taxon_name: Name of the taxon (family, genus, or species)
            rank: The rank of the taxon ('family', 'genus', 'species')
            per_page: Number of observations to fetch per page
            page: Page number to fetch

        Returns:
            Dict containing observations or None if request fails
        """
        params = {
            "taxon_name": taxon_name,
            "rank": rank,
            "per_page": per_page,
            "page": page,
            "photos": True,
        }

        try:
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            time.sleep(self.RATE_LIMIT_DELAY)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from iNaturalist: {e}")
            return None

    def download_image(self, url: str, save_path: Union[str, Path]) -> bool:
        """
        Download an image from a URL and save it to the specified path.

        Args:
            url: URL of the image to download
            save_path: Local path to save the image

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(save_path, "wb") as file,
                tqdm(
                    desc=f"Downloading {os.path.basename(save_path)}",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar,
            ):
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        size = file.write(chunk)
                        progress_bar.update(size)

            logger.info(f"Image downloaded: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return False

    def download_taxon_images(
        self,
        taxon_name: str,
        rank: str,
        download_dir: Union[str, Path],
        max_images: int = 10,
    ) -> int:
        """
        Download images of a specific taxon from iNaturalist.

        Args:
            taxon_name: Name of the taxon
            rank: Rank of the taxon ('family', 'genus', 'species')
            download_dir: Directory to save the downloaded images
            max_images: Maximum number of images to download

        Returns:
            int: Number of successfully downloaded images
        """
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        page = 1
        downloaded_count = 0

        while downloaded_count < max_images:
            observations = self.fetch_observations(taxon_name, rank, page=page)
            if not observations or not observations.get("results"):
                logger.info("No more observations found.")
                break

            for result in observations["results"]:
                if downloaded_count >= max_images:
                    break

                for photo in result.get("photos", []):
                    if downloaded_count >= max_images:
                        break

                    image_url = photo.get("url")
                    if image_url:
                        # Construct the URL for the original-sized image
                        original_url = image_url.replace("square", "original")
                        image_id = photo.get("id")
                        extension = original_url.split(".")[-1]
                        save_path = (
                            download_dir / f"{taxon_name}_{image_id}.{extension}"
                        )

                        if self.download_image(original_url, save_path):
                            downloaded_count += 1
                            time.sleep(self.RATE_LIMIT_DELAY)  # Rate limiting

            page += 1

        logger.info(
            f"Downloaded {downloaded_count} images for taxon '{taxon_name}' with rank '{rank}'."
        )
        return downloaded_count


def main():
    """Main function to demonstrate usage."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    scraper = INaturalistScraper()
    scraper.download_taxon_images(
        taxon_name="Gobiidae",
        rank="family",
        download_dir="data/gobiidae",
        max_images=10,
    )


if __name__ == "__main__":
    main()
