import optuna
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from flexible_inception_net import FlexibleInceptionNet
from flexible_inception_net import FlexibleInceptionNet
from hourglass import StackedHourglass
from inception_net_config import inception_config1
import numpy as np
import random

def objective_hourglass_net(trial, train_loader, valid_loader, train_func, device, seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
    # Suggest values for the hyperparameters
    lambda_reg = trial.suggest_float('lambda_reg', 1e-6, 1e-3, log=True)
    main_lr = trial.suggest_float('main_lr', 1e-5, 1e-3, log=True)
    stn_lr = trial.suggest_float('stn_lr', 1e-6, 1e-4, log=True)
    aux_lr = trial.suggest_float('aux_lr', 1e-5, 1e-3, log=True)
    max_norm = trial.suggest_categorical('max_norm', [None, 1.0, 5.0])
    weight_decay_stn = trial.suggest_float('weight_decay_stn', 1e-6, 1e-2, log=True)
    weight_decay_main = trial.suggest_float('weight_decay_main', 1e-6, 1e-2, log=True)
    weight_decay_aux = trial.suggest_float('weight_decay_aux', 1e-6, 1e-2, log=True)
    
    use_aux = trial.suggest_categorical('use_aux', [True, False])
    use_spatial_transformer = trial.suggest_categorical('use_spatial_transformer', [True, False])

    #depth = trial.suggest_int('depth', 1, 4, log=True)
    #stacks = trial.suggest_int('stacks', 1, 4, log=True)
    #num_channels = trial.suggest_categorical("num_channels", [16, 32, 64])
    
    # Instantiate the model with the sampled hyperparameters
    model = StackedHourglass(
        num_keypoints=68,
        num_channels=64,
        depth=4,
        stacks=2,
        spatial_transform=use_spatial_transformer,        
        use_aux_classifier=use_aux
    )

    model.to(device)

    # Define the criterion
    criterion = nn.SmoothL1Loss()

    # Extract parameters (as in your current code)
    if hasattr(model, 'stn'):
        stn_params = list(model.stn.parameters())
        stn_param_ids = list(map(id, stn_params))
    else:
        stn_params = []
        stn_param_ids = []

    if hasattr(model, 'auxiliary_classifiers'):
        aux_params = list(model.auxiliary_classifiers.parameters())
        aux_param_ids = list(map(id, aux_params))
    else:
        aux_params = []
        aux_param_ids = []

    rest_params = [
        param for param in model.parameters()
        if id(param) not in stn_param_ids and id(param) not in aux_param_ids
    ]

    # Define the optimizer based on the sampled optimizer name    
    optimizer_class = optim.Adam
    optimizer_params = {}

    optimizer = optimizer_class([
        {'params': rest_params, 'weight_decay': weight_decay_main, 'lr': main_lr, **optimizer_params},
        {'params': stn_params, 'weight_decay': weight_decay_stn, 'lr': stn_lr, **optimizer_params},
        {'params': aux_params, 'weight_decay': weight_decay_aux, 'lr': aux_lr, **optimizer_params}
    ])

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    # Set number of epochs to a smaller value for hyperparameter tuning
    n_epochs = 15
    patience = 5

    # Train the model
    train_losses, val_losses, best_loss = train_func(
        model, n_epochs, optimizer, criterion,
        train_loader=train_loader, valid_loader=valid_loader,
        patience=patience, lr_scheduler=lr_scheduler,
        lambda_reg=lambda_reg, max_norm_gradient_clipping=max_norm,
        logging=False  # Disable logging during hyperparameter tuning
    )

    # Return the validation loss of the last epoch
    val_loss = best_loss if best_loss else float('inf')

    return val_loss
