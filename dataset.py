import os
from pathlib import Path
import zipfile
import requests

def setup_local(root_path):
    root_path = Path(root_path)
    root_image_path = root_path / "dogvscat"

    if not root_image_path.is_dir():
        print(f'Creating directory {root_image_path}...')
        root_image_path.mkdir(parents=True, exist_ok=True)
    return root_path, root_image_path

def download_data(root_path='./data', zipfile_name='dogvscat.zip'):
    root_path, root_image_path = setup_local(root_path)
    zipfile_path = root_path / zipfile_name
    # Download data
    request = requests.get('https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip')
    with open(zipfile_path, 'wb') as f:
        print(f'Downloading {zipfile_path}...')
        f.write(request.content)

    # Unzip data
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        print(f'Unzipping {zipfile_path}...')
        zip_ref.extractall(root_image_path)
