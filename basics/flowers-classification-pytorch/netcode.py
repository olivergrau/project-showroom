import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
import time
import os

def initialize_model(arch, class_to_idx, hidden_units, learning_rate, gpu):
    """
    Initializes a model based on the specified architecture, number of hidden units, learning rate, and device (CPU/GPU).

    Args:
        arch (str): The architecture of the model ('resnet18' or 'vgg13').
        class_to_idx (dict): A mapping from class names to indices.
        hidden_units (int): The number of hidden units in the custom layer.
        learning_rate (float): The learning rate for the optimizer.
        gpu (bool): If True, use GPU for training.

    Returns:
        model (nn.Module): The initialized model.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Adam): The optimizer.
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Model selection based on architecture
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.arch = arch

        num_ftrs = model.fc.in_features

        if hidden_units > 0:
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, hidden_units),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_units, len(class_to_idx))
            )
        else:
            model.fc = nn.Linear(num_ftrs, len(class_to_idx))

    elif arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features

        # Adding custom hidden units decreases the performance because the new layer has to be trained from scratch
        if hidden_units > 0:
            classifier = list(model.classifier.children())
            classifier[6] = nn.Linear(num_ftrs, hidden_units)
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(0.2))
            classifier.append(nn.Linear(hidden_units, len(class_to_idx)))
            model.classifier = nn.Sequential(*classifier)
        else:
            model.classifier[6] = nn.Linear(num_ftrs, len(class_to_idx))
    else:
        raise ValueError("Unsupported architecture. Please choose 'resnet18' or 'vgg13'.")

    model.class_to_idx = class_to_idx

    # Only the classification layer should be trained
    for param in model.parameters():
        param.requires_grad = False

    if arch == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == 'vgg13':
        for param in model.classifier[6].parameters():
            param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    return model, criterion, optimizer

def load_checkpoint(checkpoint_path):
    """
    Loads a checkpoint and restores the model, optimizer, and loss function states.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        model (nn.Module): The restored model.
        optimizer (optim.Adam): The restored optimizer.
        criterion (nn.CrossEntropyLoss): The restored loss function.
        epoch (int): The epoch to resume training from.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if checkpoint['arch'] == "resnet18":
            model = models.__dict__[checkpoint['arch']](weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.__dict__[checkpoint['arch']](weights=models.VGG13_Weights.DEFAULT)

        if checkpoint['arch'] == 'resnet18':
            model.fc = checkpoint['classifier']
        else:
            model.classifier = checkpoint['classifier']

        model.arch = checkpoint['arch']

        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        # The lambda is needed for the Adam optimzier because of the selective gradient activation
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=checkpoint['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        criterion.load_state_dict(checkpoint['criterion'])

        # Only the classification layer should be trained
        for param in model.parameters():
            param.requires_grad = False

        if checkpoint['arch'] == 'resnet18':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif checkpoint['arch'] == 'vgg13':
            for param in model.classifier[6].parameters():
                param.requires_grad = True

        return model, optimizer, criterion, int(checkpoint['epoch']) + 1
    else:
        raise ValueError("Checkpoint doesn't exist.")

def save_checkpoint(model, optimizer, save_dir, epoch, arch, criterion, class_to_idx, learning_rate):
    """
    Saves the current state of the model, optimizer, and other relevant information to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Adam): The optimizer to save.
        save_dir (str): The directory to save the checkpoint file in.
        epoch (int): The current epoch.
        arch (str): The model architecture.
        criterion (nn.CrossEntropyLoss): The loss function to save.
        class_to_idx (dict): A mapping from class names to indices.
        learning_rate (float): The learning rate.
    """
    checkpoint = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.fc if arch == 'resnet18' else model.classifier,
        'class_to_idx': class_to_idx,
        'criterion': criterion.state_dict(),
        'learning_rate': learning_rate
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, save_dir, start_epoch=0, num_epochs=20, learning_rate=0.1):
    """
    Trains the model and saves the best model weights.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Adam): The optimizer.
        scheduler (optim.lr_scheduler): The learning rate scheduler.
        dataloaders (dict): The dataloaders for training and validation data.
        dataset_sizes (dict): The sizes of the training and validation datasets.
        device (torch.device): The device to train on (CPU/GPU).
        save_dir (str): The directory to save checkpoints.
        start_epoch (int): The epoch to start training from.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate.

    Returns:
        model (nn.Module): The trained model with the best weights.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_since = time.time()

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = time.time() - epoch_since
        print(f'Epoch {epoch} complete in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')
        print()

        # Save model after every epoch
        if save_dir:
            save_checkpoint(model, optimizer, save_dir, epoch, model.arch, criterion, model.class_to_idx, learning_rate)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the test set.
        criterion (nn.CrossEntropyLoss): The loss function.
        device (torch.device): The device to evaluate on (CPU/GPU).

    Returns:
        None
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')
