import torch 
from torch.utils.data import Dataset 
from torchvision import transforms 
from PIL import Image 
import os 
from pathlib import Path

class DogsvsCatsDataset(Dataset):
    def __init__(self, tar_dir: str, transform: transforms=None):
        self.image_paths = self._walk_through_dir(tar_dir)
        self.classes, self.cls_to_idx = self._find_classes(tar_dir)
        self.transform = transform 
        
    def _walk_through_dir(self,tar_dir:str):
        dir_paths = [dir_path for dir_path, _, file_names in os.walk(tar_dir) if file_names]
        image_paths = []
        for path in dir_paths:
            image_paths += Path(path).glob("*.jpg")
        return image_paths
    
    def find_classes(self,tar_dir):
        classes = sorted(entry.name for entry in os.scandir(tar_dir) if entry.is_dir())
        if not classes:
            raise FileNotFoundError("Counld't find any class name, please check your target directory!")
        cls_to_idx = {cls:id for id, cls in enumerate(classes)}
        return classes, cls_to_idx 
    
    def _load_images(self, index):
        image = self.image_paths[index]
        return Image.open(image)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = self._load_images(index)
        classes = self.image_paths[index].parent.stem
        cls_to_idx = self.cls_to_idx[classes]
        if self.transform:
            image = self.transform(image)
        return image, cls_to_idx