import torch
import torch.nn as nn
from torchvision import models
import torchvision

def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer = getattr(models, model_name)(pretrained=True)
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Get the number of input features for the classifier
    num_ftrs = model_transfer.fc.in_features

    # Replace the fully connected layer with a more complex classification head
    import torch.nn as nn

    # Replace the fully connected layer with a more complex classification head
    model_transfer.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),           # First linear layer with size 1024
        nn.BatchNorm1d(1024),                # Batch Normalization
        nn.LeakyReLU(0.1),                   # Leaky ReLU for better Learning with negative values
        nn.Dropout(0.5),                     # Dropout with 50% for regularization
        nn.Linear(1024, 512),                # Another fully connected layer
        nn.BatchNorm1d(512),                 # Batch Normalization for stablelization
        nn.LeakyReLU(0.1),                   # Leaky ReLU
        nn.Dropout(0.5),                     # Dropout
        nn.Linear(512, n_classes)            # Final classification layer
    )

    return model_transfer




######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
