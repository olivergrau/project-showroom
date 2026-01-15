import os
import json
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List

from utils.model import *


# Implement the student model
# Make sure to parametrize the __init__() correctly
class MobileNetV3_Household_Small(nn.Module):
    """
    Student model based on MobileNetV3-Small for the household objects dataset.

    This model mirrors the teacher's general structure but reduces width and
    classifier size to create a smaller network suitable for knowledge distillation.

    Args:
        num_classes (int): Number of output classes (default: 10).
        width_mult (float): Width multiplier for MobileNetV3 backbone
                            (controls channel counts, default: 0.6).
        linear_size (int): Hidden dimension size of the classifier MLP
                           (default: 256).
        dropout (float): Dropout probability in the classifier (default: 0.2).
    """

    def __init__(self, num_classes: int = 10,
                 width_mult: float = 0.6,
                 linear_size: int = 256,
                 dropout: float = 0.2) -> None:
        super().__init__()

        # Store hyperparameters so we can reconstruct this model later when
        # loading from a checkpoint
        self.num_classes = num_classes
        self.width_mult = float(width_mult)
        self.linear_size = int(linear_size)
        self.dropout = float(dropout)

        # Load a MobileNetV3-Small backbone with ImageNet pretrained weights
        # and reduced width. This preserves the inductive bias but cuts
        # parameter count.
        self.backbone = models.mobilenet_v3_small(            
            width_mult=self.width_mult,
            weights=None  # No pretrained weights for reduced width
        )

        # Determine the feature dimension coming out of the backbone
        # (before the classifier). For torchvision's MobileNetV3 this is
        # the in_features of the first classifier layer.
        last_channel = self.backbone.classifier[0].in_features

        # Replace the original classifier with a smaller MLP head suitable
        # for our 10-class household objects task.
        self.backbone.classifier = nn.Sequential(
            nn.Linear(last_channel, self.linear_size),
            nn.Hardswish(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.linear_size, self.num_classes),
        )

        # For compatibility with existing code that expects self.model
        # (like in the teacher implementation), expose the backbone as model.
        self.model = self.backbone

    def forward(self, x):
        # Ensure input is correctly sized (CIFAR-like 32x32 → 224x224)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(x)

    
    
# Implement the logic to compute the knowledge distillation loss
# Remember that temperature is a weight for the teacher's probability distribution and 
# alpha is the weight balancing teacher loss vs student loss
def _knowledge_distillation_loss(student_logits, teacher_logits, targets,
                                 temperature: float = 2.0,
                                 alpha: float = 0.5):
    """
    Compute the knowledge distillation loss.

    Args:
        student_logits: Output logits from the student model (batch_size x num_classes)
        teacher_logits: Output logits from the teacher model (batch_size x num_classes)
        targets: Ground truth labels (batch_size,)
        temperature: Temperature for softening the teacher's probability distribution
        alpha: Weight balancing distillation loss vs hard cross entropy loss

    Returns:
        Final loss combining distillation and standard cross entropy.
    """
    # Hard loss: standard cross entropy with ground truth labels
    hard_loss = F.cross_entropy(student_logits, targets)

    # Softened distributions for KD
    T = temperature

    # Student log probabilities at temperature T
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    # Teacher probabilities at temperature T (teacher is treated as fixed target distribution)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)

    # KL divergence between teacher and student distributions
    # Using batchmean reduction to match the standard KD formulation
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    distillation_loss = kl_loss_fn(student_log_probs, teacher_probs) * (T * T)

    # Combine both parts:
    # alpha controls how much we trust the teacher (distillation),
    # (1 - alpha) keeps some weight on the original ground-truth labels.
    loss = alpha * distillation_loss + (1.0 - alpha) * hard_loss
    return loss

def _distill_single_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
    grad_clip_norm: Optional[float] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None
) -> Tuple[float, float]:
    """
    Train a student model for a single epoch using knowledge distillation.
    
    Args:
        student_model: Smaller model to be trained (student)
        teacher_model: Larger pre-trained model (teacher)
        train_loader: Training data loader
        criterion: Standard loss function (for hard targets)  # TODO: kept for API symmetry, not used directly here
        optimizer: PyTorch optimizer for student model
        device: Device to train on
        temperature: Temperature for softening probability distributions
        alpha: Weight for balancing hard and soft targets
        grad_clip_norm: Maximum norm for gradient clipping (optional)
        epoch: Current epoch (optional, for progress display)
        num_epochs: Total number of epochs (optional, for progress display)
        
    Returns:
        Tuple of (train_loss, train_accuracy)
    """
    # Set models to appropriate modes
    student_model.train()
    teacher_model.eval()  # Teacher is always in eval mode
    
    # Initialize tracking variables
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Progress bar description
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs} [Distill]"
    else:
        desc = "Distillation"
        
    pbar = tqdm(train_loader, desc=desc)
    
    # Iterate over batches
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero out previous gradients
        optimizer.zero_grad()

        # Forward pass: student with gradients
        student_outputs = student_model(inputs)

        # Forward pass: teacher without gradients
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Compute distillation loss
        loss = _knowledge_distillation_loss(
            student_outputs,
            teacher_outputs,
            targets,
            temperature=temperature,
            alpha=alpha
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip_norm)
            
        # Update weights
        optimizer.step()
        
        # Update statistics
        batch_size = inputs.size(0)
        train_loss += loss.item() * batch_size
        
        # Update accuracy metrics (based on hard predictions)
        _, predicted = student_outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
            
        train_correct += batch_correct
        train_total += batch_total
        
        # Update progress bar with current batch stats
        pbar.set_postfix({
            "loss": loss.item(),
            "batch_acc": 100.0 * batch_correct / batch_total,
            "running_acc": 100.0 * train_correct / train_total,
            "lr": optimizer.param_groups[0]['lr']
        })
    
    # Calculate final metrics for training - average over all samples
    train_loss = train_loss / train_total
    train_accuracy = 100.0 * train_correct / train_total
    
    return train_loss, train_accuracy


def train_with_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    test_loader,
    training_config: Dict[str, Any],
    checkpoint_path: str = 'checkpoints/distilled_model.pth'
) -> Tuple[nn.Module, Dict[str, List], float, int]:
    """
    Train a student model using knowledge distillation from a teacher model.
    
    Args:
        student_model: Smaller model to be trained (student)
        teacher_model: Larger pre-trained model (teacher)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration
        checkpoint_path: Path to save the best student model
        
    Returns:
        Tuple of (student_model, training_stats, best_accuracy, best_epoch)
    """
    # Extract training configuration
    num_epochs = training_config.get('num_epochs')
    standard_criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience')
    device = training_config.get('device', 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    
    # Distillation specific parameters
    temperature = training_config.get('temperature')
    alpha = training_config.get('alpha')
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    # Display training configuration
    print(f"Student parameters: {count_parameters(student_model):,}")
    print(f"Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"Training with knowledge distillation for {num_epochs} epochs")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    
    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    training_stats = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_time": [],
        "lr": []
    }
    
    # Early stopping variable
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch with distillation
        train_loss, train_accuracy = _distill_single_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            criterion=standard_criterion,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            alpha=alpha,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        
        # Evaluate student on test set (standard evaluation, no distillation)
        test_loss, test_accuracy = validate_single_epoch(
            student_model, test_loader, standard_criterion, device, epoch, num_epochs
        )
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print statistics
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
              f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model
        if test_accuracy > best_accuracy:
            print(f"New best student model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            
            save_model(student_model, checkpoint_path)
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1
        
        # Early stopping condition
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
        
        # Record statistics
        training_stats["epoch"].append(epoch + 1)
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        training_stats["test_loss"].append(test_loss)
        training_stats["test_accuracy"].append(test_accuracy)
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)
    
    # Load the best student model
    student_model = load_model(
        checkpoint_path, device, 
        model_class=MobileNetV3_Household_Small, 
        width_mult=student_model.width_mult, 
        linear_size=student_model.linear_size, 
        dropout=student_model.dropout)
    
    print(f"Distillation completed. Best student accuracy: {best_accuracy:.2f}%")
    print(f"Best student model saved as '{checkpoint_path}' at epoch {best_epoch}")
    
    return student_model, training_stats, best_accuracy, best_epoch