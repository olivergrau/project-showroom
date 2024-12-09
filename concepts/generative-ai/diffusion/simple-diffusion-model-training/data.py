from torch.utils.data import ConcatDataset
from multiprocessing import cpu_count
import multiprocessing
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class CompCarsDataset(Dataset):
    def __init__(self, root_path, annotations_file, transform=None):
        """
        Custom Dataset class for CompCars dataset.

        Args:
            root_path (str): Path to the root directory of CompCars dataset.
            annotations_file (str): Path to the annotation file.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_path = root_path
        self.transform = transform

        # Load image paths from annotations file
        with open(annotations_file, 'r') as f:
            lines = f.readlines()

        self.image_paths = [os.path.join(root_path, "image/" + line.split()[0]) for line in lines]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path
        img_path = self.image_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(img_path)  # Load image in BGR format
        if image is None:
            raise RuntimeError(f"Failed to load image at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL image
        image = Image.fromarray(image)

        # Apply transformations (ensure it expects an RGB image)
        if self.transform:
            image = self.transform(image)

        return image

def get_compcars_gan_dataloader(root_path, annotations_files, image_size, batch_size, workers=4):
    """
    Creates a DataLoader for CompCars dataset for GANs.

    Args:
        root_path (str): Path to the CompCars dataset.
        annotations_files (str): Path to the annotation file for the dataset split (multiple).
        image_size (int): Size to resize the images.
        batch_size (int): Batch size for the DataLoader.
        workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader: DataLoader for CompCars dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),   
            transforms.RandomHorizontalFlip(),         
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    datasets = []
    for annotations_file in annotations_files:
        dataset = CompCarsDataset(
            root_path=root_path,
            annotations_file=annotations_file,
            transform=transform
        )
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )

    return dataloader

