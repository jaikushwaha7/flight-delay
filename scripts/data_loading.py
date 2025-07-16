import os
import logging
import subprocess

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataDownload:
    """
    Handles downloading data from Kaggle.
    """
    def __init__(self, dataset_name: str, download_path: str = "data/raw"):
        self.dataset_name = dataset_name
        self.download_path = download_path
        self.is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        logger.info(f"DataDownload initialized. Running in Colab: {self.is_colab}")
        os.makedirs(self.download_path, exist_ok=True)

    def _setup_kaggle_colab(self):
        """
        Sets up Kaggle API in Google Colab.
        Assumes kaggle.json is uploaded to /content/ drive or available.
        """
        if not self.is_colab:
            return

        logger.info("Setting up Kaggle API for Colab environment...")
        # Check if kaggle.json exists, if not, prompt user
        if not os.path.exists('/root/.kaggle/kaggle.json'):
            logger.warning("kaggle.json not found at /root/.kaggle/. Please upload it to your Colab session.")
            logger.warning("Follow these steps:")
            logger.warning("1. Go to your Kaggle account settings -> 'API' section -> 'Create New API Token'.")
            logger.warning("2. This will download kaggle.json.")
            logger.warning("3. In Colab, click 'Files' (folder icon on left sidebar) -> 'Mount Drive' (if not already mounted) OR upload kaggle.json directly.")
            logger.warning("   If uploading directly, you might need to manually move it: !mkdir -p ~/.kaggle/ && !cp kaggle.json ~/.kaggle/ && !chmod 600 ~/.kaggle/kaggle.json")
            # For simplicity, we assume the user has handled this or uploaded it manually
            # A more robust solution might wait for user input or auto-upload from drive.

        # Ensure correct permissions
        os.makedirs('/root/.kaggle', exist_ok=True)
        # Assuming kaggle.json is placed in the right spot by user (e.g., via manual upload or drive)
        # If it's uploaded to /content/kaggle.json
        if os.path.exists('/content/kaggle.json') and not os.path.exists('/root/.kaggle/kaggle.json'):
            subprocess.run(['mv', '/content/kaggle.json', '/root/.kaggle/kaggle.json'], check=True)
            subprocess.run(['chmod', '600', '/root/.kaggle/kaggle.json'], check=True)
            logger.info("Moved kaggle.json to ~/.kaggle/ and set permissions.")
        elif not os.path.exists('/root/.kaggle/kaggle.json'):
            logger.critical("Kaggle API key (kaggle.json) not found. Please upload it and restart the process.")
            raise FileNotFoundError("Kaggle API key 'kaggle.json' not found.")


    def data_download(self) -> str:
        """
        Downloads the specified Kaggle dataset to the download_path.

        Returns:
            str: The path to the directory where the dataset is downloaded.
        """
        if self.is_colab:
            self._setup_kaggle_colab()

        # Check if data already exists to avoid re-downloading
        dataset_dir_name = self.dataset_name.split('/')[-1].replace('-', '_')
        expected_data_path = os.path.join(self.download_path, dataset_dir_name)

        # Check for presence of any CSV file inside the expected_data_path
        # This is a more robust check than just the directory existing, as directory might be empty.
        data_found = False
        if os.path.exists(expected_data_path):
            for root, _, files in os.walk(expected_data_path):
                if any(f.endswith('.csv') for f in files):
                    data_found = True
                    break
        
        if data_found:
            logger.info(f"Dataset already appears to be downloaded and unzipped at: {expected_data_path}")
            return expected_data_path
        else:
            logger.info(f"Downloading dataset '{self.dataset_name}' to '{self.download_path}'...")
            try:
                # Use Kaggle API to download
                subprocess.run(['kaggle', 'datasets', 'download', '-d', self.dataset_name, '-p', self.download_path, '--unzip'], check=True)
                logger.info("Dataset downloaded and unzipped successfully!")
                
                # Kaggle downloads into a folder named after the dataset's last part (e.g., flight-delay-and-cancellation-data-2019-2023-v2)
                # It does NOT create the dataset_dir_name based on replace('-', '_') automatically.
                # We need to find the actual unzipped directory and return its path.
                
                # List contents of download_path to find the unzipped directory
                downloaded_contents = os.listdir(self.download_path)
                
                actual_dataset_dir = None
                for item in downloaded_contents:
                    full_path = os.path.join(self.download_path, item)
                    if os.path.isdir(full_path) and 'flights_sample_100k.csv' in os.listdir(full_path):
                        actual_dataset_dir = full_path
                        break
                
                if actual_dataset_dir:
                    logger.info(f"Actual dataset directory found at: {actual_dataset_dir}")
                    return actual_dataset_dir
                else:
                    logger.error(f"Could not find the unzipped dataset directory containing 'flights_sample_100k.csv' in {self.download_path}.")
                    # Fallback to the constructed name, though it might not be accurate if Kaggle uses a different folder name
                    return expected_data_path 

            except subprocess.CalledProcessError as e:
                logger.error(f"Kaggle download failed: {e}")
                logger.error("Please ensure Kaggle API is correctly configured (kaggle.json with right permissions).")
                raise
            except FileNotFoundError:
                logger.error("Kaggle command not found. Please install the Kaggle API client: `pip install kaggle`")
                raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during data download: {e}", exc_info=True)
                raise