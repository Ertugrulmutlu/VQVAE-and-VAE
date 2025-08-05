from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
from torch import float
from PIL import Image

class DataPreprocessor:
    def __init__(self, 
                 dataset_path: str = "vesikalik_dataset", 
                 image_size: int = 64,
                 batch_size: int = 64,
                 num_workers: int = 2,
                 shuffle: bool = True):
        """
        Initializes the preprocessor with settings for resizing, batching, and loading.

        Args:
            dataset_path (str): Path to dataset directory (must contain at least one subfolder).
            image_size (int): Size to which all images will be resized.
            batch_size (int): Number of images per training batch.
            num_workers (int): Number of parallel workers for loading data.
            shuffle (bool): Whether to shuffle the dataset each epoch.
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.Lambda(self.convert_to_rgb),
            transforms.Resize((128, 128), interpolation=Image.LANCZOS),  # upscale
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),  # downscale
            transforms.ToTensor(),
        ])

        # Safety check: must contain at least one subfolder
        if not any(self.dataset_path.iterdir()):
            raise RuntimeError(f"Dataset folder '{self.dataset_path}' is empty or improperly structured!")
    def convert_to_rgb(self,img):
        return img.convert("RGB")
    def get_loader(self) -> Tuple[DataLoader, int]:
        """
        Returns a DataLoader for the dataset along with the number of channels.

        Returns:
            Tuple[DataLoader, int]: The DataLoader and number of input channels (3 for RGB).
        """
        dataset = datasets.ImageFolder(root=str(self.dataset_path), transform=self.transform)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers)
        return dataloader, 3
