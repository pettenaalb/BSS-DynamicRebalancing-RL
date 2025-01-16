import os
import platform
import requests
import zipfile

data_path = "../data/"
if platform.system() == "Linux":
    data_path = "/mnt/mydisk/edoardo_scarpel/data/"

# ----------------------------------------------------------------------------------------------------------------------

def download_and_extract(url, save_path, extract_to):
    """
    Downloads a file from the given URL, saves it to the specified path,
    extracts it if it's a ZIP file, and removes the ZIP file after extraction.

    :param url: The URL to download the file from.
    :param save_path: The local path to save the downloaded file.
    :param extract_to: The directory to extract the contents to.
    """
    try:
        # Download the file
        print(f"Downloading file from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")

        # Check if the file is a ZIP file and extract it
        if zipfile.is_zipfile(save_path):
            print(f"Extracting contents of {save_path} to {extract_to}...")
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extraction complete. Files extracted to {extract_to}")

            # Remove the ZIP file
            os.remove(save_path)
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

    for month in range(0, 12):
        url = 'https://s3.amazonaws.com/hubway-data/2022' + str(month).zfill(2) + '-bluebikes-tripdata.zip'
        download_and_extract(url, save_path, save_path)

    os.remove(save_path + '__MACOSX/')  # Remove the __MACOSX directory


if __name__ == '__main__':
    main()