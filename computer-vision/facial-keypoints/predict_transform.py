import cv2
import torch

# mean=0.0014, std=0.0009
def prepare_img_for_prediction(img, mean=0.0014, std=0.0010, output_size=(224, 224)):
    """
    Prepare an image for keypoint prediction by applying resizing, normalization, and tensor transformation.
    
    Args:
        img (numpy.ndarray): Input image in RGB format.
        mean (float): Mean for normalization (if applicable).
        std (float): Standard deviation for normalization (if applicable).
        output_size (tuple): Desired output size (height, width).
    
    Returns:
        torch.Tensor: Transformed image ready for model prediction, with shape (1, C, H, W).
    """

    # Resize the image to the target size
    h, w = img.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Normalize the image (convert to grayscale and scale to [0, 1])
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    img_normalized = img_gray / 255.0
    
    if mean is not None and std is not None:
        img_normalized = (img_normalized - mean) / std

    return img_normalized.reshape(1, 224, 224)        