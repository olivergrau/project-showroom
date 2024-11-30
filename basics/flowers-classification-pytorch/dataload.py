import torch
from torchvision import datasets, transforms

def load_data(data_dir='../flower_data', batch_size=64):
    """
    Loads and preprocesses the dataset, applies transformations, and returns data loaders for training, validation, and testing.

    Args:
        data_dir (str): The directory containing the dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        data_transforms (dict): Transformations applied to the data.
        image_datasets (dict): The datasets for training, validation, and testing.
        dataloaders (dict): Data loaders for the datasets.
        dataset_sizes (dict): The sizes of the datasets.
        class_names (list): The names of the classes.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Training data augmentation and normalization
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # data normalization
    ])

    # Validation and testing data normalization
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # data normalization
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # data normalization
    ])

    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    data_transforms = {'train': train_transforms, 'test': test_transforms, 'valid': valid_transforms}
    image_datasets = {'train': train_data, 'test': test_data, 'valid': valid_data}
    dataloaders = {'train': trainloader, 'test': testloader, 'valid': validloader}

    dataset_sizes = {
        'train': len(image_datasets['train']),
        'valid': len(image_datasets['valid']),
        'test': len(image_datasets['test'])
    }

    class_names = image_datasets['train'].classes

    return data_transforms, image_datasets, dataloaders, dataset_sizes, class_names
