#%%
import os
import requests
import gzip
import shutil
import io
import zipfile
import tarfile
import urllib.request


def download_and_extract_zip(url: str, output_dir: str = './', extract_dir: str = 'extracted') -> str:
    """
    Function to download a zip file from a URL, extract it, and save it to a new directory.

    Args:
        url (str): URL of the file to download.
        output_dir (str): Path to the parent directory where the extracted files will be saved.
        extract_dir (str): Name of the directory where the extracted files will be saved.

    Returns:
        str: Path to the directory where the extracted files are saved.
    """
    # Get the file name from the URL
    filename = url.split('/')[-1]

    # Create the path to save the downloaded file
    output_file = os.path.join(output_dir, filename)

    # Download the file from the URL
    response = requests.get(url)

    # Extract the zip file and save the files
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        # Create the directory for extraction
        extract_path = os.path.join(output_dir, extract_dir)
        os.makedirs(extract_path, exist_ok=True)
        # Extract files from the zip archive
        zip_file.extractall(extract_path)

    # Return the path to the extracted files
    return extract_path

        
#%%
if __name__ == "__main__":
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    folder = '/home1/data/common-data/COCO/'
    dir_name = 'annotations'
    os.makedirs(folder+dir_name)
    
    download_and_extract_zip(url, folder, dir_name)
# %%
