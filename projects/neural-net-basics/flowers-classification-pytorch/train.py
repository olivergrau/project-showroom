import argparse
from dataload import load_data
from netcode import initialize_model, train_model, load_checkpoint, evaluate_model
import torch
import os

def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue

    def positive_float(value):
        fvalue = float(value)
        if fvalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
        return fvalue
    
    parser = argparse.ArgumentParser(description="Train a new network on a data set")
    
    # basic usage    
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')

    # options
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints', default='')
    parser.add_argument('--arch', type=str, help='Model architecture (e.g., "resnet18, vgg13")', default='resnet18')
    parser.add_argument('--learning_rate', type=positive_float, help='Learning rate', default=0.1)
    parser.add_argument('--hidden_units', type=positive_int, help='Number of hidden units', default=-1)
    parser.add_argument('--epochs', type=positive_int, help='Number of epochs', default=5)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--show_model', action='store_true', help='Show model for debug purposes')
    
    # parse args
    args = parser.parse_args()

    # print args and options
    print(f"Data directory: {args.data_directory}")
    print(f"Save directory: {args.save_dir}")
    print(f"Architecture: {args.arch}")
    print(f"Learning rate: {args.learning_rate}")
    
    if args.hidden_units == -1:
        print(f"Default classifier of selected net will be used")
    else:
        print(f"Modify classifier to integrate a fc with hidden units: {args.hidden_units}")
        
    print(f"Epochs: {args.epochs}")
    print(f"Use GPU: {args.gpu}")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # load data
    data_transforms, image_datasets, dataloaders, dataset_sizes, class_names = load_data(args.data_directory)

    # Information to the loaded data
    print("\nData Transforms:")
    for k, v in data_transforms.items():
        print(f"  {k}: {v}")

    print("\nDataset Sizes:")
    for k, v in dataset_sizes.items():
        print(f"  {k}: {v} images")

    print(f"\nClasses: {len(class_names)}\n")    
    
    # load checkpoint if exists
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')    
        
    start_epoch = 0
    if args.save_dir and os.path.exists(checkpoint_path):
        model, optimizer, criterion, start_epoch = load_checkpoint(checkpoint_path)
        model = model.to(device)
        print(f"Continuing from checkpoint with epoch: {start_epoch}")
    else:
        print(f"Start training from scratch.")

        model, criterion, optimizer = initialize_model(
            args.arch, image_datasets['train'].class_to_idx, args.hidden_units, args.learning_rate, args.gpu)
    
    print()
    if args.gpu:
        print("Use GPU for training")
    
    print()    
    
    if args.show_model:
        print(model)

    # train the model
    model = train_model(
        model, criterion, optimizer, None, dataloaders, dataset_sizes, device, args.save_dir, start_epoch, args.epochs + start_epoch, learning_rate=args.learning_rate)

    # evaluating the model
    print("\nEvaluating the model on the test set...")
    evaluate_model(model, dataloaders['test'], criterion, device)
    
if __name__ == "__main__":
    main()