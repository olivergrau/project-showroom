import argparse
import json
import torch
from PIL import Image
import numpy as np
from netcode import load_checkpoint

def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    image = Image.open(image_path)

    # Scale the image to shortest side and keep aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        new_size = (int(aspect_ratio * 256), 256)
    else:
        new_size = (256, int(256 / aspect_ratio))

    image = image.resize(new_size)

    # crop around the center
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    # get a numpy array
    np_image = np.array(image)

    # normalize to 0-1
    np_image = np_image / 255.0

    # mean and deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # dimension ordering: color at first place
    np_image = np_image.transpose((2, 0, 1))

    return torch.from_numpy(np_image).float()

def predict(image_path, model, device, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    loaded_img = process_image(image_path)
    loaded_img = loaded_img.to(device)  # move the img to the device

    model.eval()

    with torch.no_grad():
        loaded_img = loaded_img.unsqueeze(0)  # add batch dimension because model expects batch
        output = model(loaded_img)
        probs, indices = torch.topk(output, topk)
        probs = torch.nn.functional.softmax(probs, dim=1)  # Apply softmax to get probabilities

    # Indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices[0].cpu().numpy()]

    return probs[0].cpu().numpy(), classes

def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
        return ivalue
    
    parser = argparse.ArgumentParser(description="Predict class of a provided image")

    # basic usage    
    parser.add_argument('image_path', type=str, help='Path to the image which should be predicted')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint info for trained net')

    # options        
    parser.add_argument('--top_k', type=positive_int, help='Top k classes which are predicted', default=5)
    parser.add_argument('--category_names', type=str, help='Category mapping', default=None)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')

    # parse args
    args = parser.parse_args()

    # print args and options
    print(f"Path to image: {args.image_path}")
    print(f"Path to checkpoint file: {args.checkpoint}")
    print(f"Top k classes: {args.top_k}")
    print(f"Path to category file: {args.category_names}")
    print(f"Use GPU: {args.gpu}")

    # Load the model
    model, _, _, _ = load_checkpoint(args.checkpoint)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predict
    probs, classes = predict(args.image_path, model, device, args.top_k)

    # Print results    
    print(f"Top {args.top_k} class directories: {classes}")
    
    if args.category_names is not None:
        # Load category names
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(cls)] for cls in classes]        
        print(f"Top {args.top_k} classes: {class_names}")
    
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
