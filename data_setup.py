import os
from pathlib import Path
import zipfile
import requests
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from dogsvscatsCustomDataset import DogsvsCatsDataset

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
        
    # Move data is extracted to the root_image_path
    # from root_image_path/cats_and_dogs_filtered/train and root_image_path/cats_and_dogs_filtered/test to root_image_path/train and root_image_path/test
    train_path = root_image_path / 'train'
    validation_path = root_image_path / 'validation'
    shutil.move(root_image_path / 'cats_and_dogs_filtered/train', train_path)
    shutil.move(root_image_path / 'cats_and_dogs_filtered/validation', validation_path)
    # remove root_image_path / 'cats_and_dogs_filtered/'
    shutil.rmtree(root_image_path / 'cats_and_dogs_filtered')
    print('Data downloaded and extracted successfully.')
    return train_path, validation_path

def create_dataloader(train_dir:str, valid_dir:str,
                      transform: transforms.Compose,
                      batch_size: int=32,
                      num_workers:int=0):
    train_data = DogsvsCatsDataset(train_dir, transform)
    valid_data = DogsvsCatsDataset(valid_dir, transform)
    
    # Get class names 
    class_names = train_data.classes 
    
    # Turn data to dataloader
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return train_dataloader, valid_dataloader, class_names