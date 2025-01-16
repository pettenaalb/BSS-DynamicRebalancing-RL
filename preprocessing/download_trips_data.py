import os
import platform
import requests
import zipfile
from urllib.parse import urlparse
from tqdm import tqdm

data_path = "../data/"
if platform.system() == "Linux":
    data_path = "/mnt/mydisk/edoardo_scarpel/data/"

# ----------------------------------------------------------------------------------------------------------------------

def download_and_extract(url, target_directory, tbar = None):
    """
    Downloads a file from the given URL, saves it to the specified directory,
    extracts it if it's a ZIP file, and removes the ZIP file after extraction.

    :param url: The URL to download the file from.
    :param target_directory: The directory where the file will be saved and extracted.
    """
    try:
        # Ensure the target directory exists
        os.makedirs(target_directory, exist_ok=True)

        # Parse the filename from the URL
        filename = os.path.basename(urlparse(url).path)
        save_path = os.path.join(target_directory, filename)

        # Download the file
        if tbar is not None:
            tbar.set_description(f"Downloading {filename}")
        else:
            print(f"Downloading file from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Check if the file is a ZIP file and extract it
        if zipfile.is_zipfile(save_path):
            if tbar is not None:
                tbar.set_description(f"Extracting {filename}")
            else:
                print(f"Extracting contents of {save_path} to {target_directory}...")
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(target_directory)

            # Remove the ZIP file
            os.remove(save_path)
            if tbar is not None:
                tbar.set_description(f"Removed the ZIP file: {save_path}")
            else:
                print(f"Removed the ZIP file: {save_path}")
        else:
            print(f"The file is not a ZIP archive. No extraction performed.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# ----------------------------------------------------------------------------------------------------------------------

def main():
    save_path = data_path + 'trips/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")

    tbar = tqdm(range(12), desc='Downloading files', position=0, leave=True)

    for month in range(0, 12):
        url = 'https://s3.amazonaws.com/hubway-data/2022' + str(month+1).zfill(2) + '-bluebikes-tripdata.zip'
        download_and_extract(url, save_path, tbar)
        tbar.update(1)

    os.remove(save_path + '__MACOSX/')  # Remove the __MACOSX directory


if __name__ == '__main__':
    main()