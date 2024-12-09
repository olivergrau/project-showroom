import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import random

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, image):
        image_copy = np.copy(image)
        
        # convert image to grayscale
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0           
        
        return image_copy

class GrayScale(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, image):
        image_copy = np.copy(image)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

        return image_copy

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
                
        return img

class RandomHorizontalFlip(object):
    """Horizontally flip the image in a sample with a given probability.

    Args:
        prob (float): Probability to apply the horizontal flip (0.0 to 1.0).
    """

    def __init__(self, prob=0.25):
        self.apply_probability = prob

    def __call__(self, image):
        # Apply flip based on probability
        if random.random() > self.apply_probability:
            return image

        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)

        return flipped_image

class FaceCrop(object):
    def __init__(self, padding=10, output_size=(224, 224)):
        self.padding = padding
        self.output_size = output_size

    def __call__(self, image):
        # Make a grayscale copy for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype(np.uint8)  # Ensure it's in 8-bit format

        # Load HAAR cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.001, minNeighbors=8, minSize=(30, 30))

        if len(faces) > 0:
            # Use the first detected face
            x, y, w, h = faces[0]
            x_min = max(0, x - self.padding)
            y_min = max(0, y - self.padding)
            x_max = min(image.shape[1], x + w + self.padding)
            y_max = min(image.shape[0], y + h + self.padding)

            # Crop the original color image
            cropped_image = image[y_min:y_max, x_min:x_max]
        else:
            # If no face is detected, use the original image
            cropped_image = image

        # Resize the cropped image to the output size
        resized_image = cv2.resize(cropped_image, self.output_size)

        return resized_image

    
class RandomRotate(object):
    """Rotate the image in a sample by a given angle in a specified percentage of cases.

    Args:
        angle (int): Rotation angle in degrees.
        prob (float): Probability to apply the rotation (0.0 to 1.0).
    """

    def __init__(self, angle, prob=0.5):
        self.angle = angle
        self.apply_probability = prob

    def __call__(self, image):
        # Randomly decide whether to apply rotation based on the apply_probability
        if random.random() > self.apply_probability:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, scale=1.0)

        # Rotate the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        return rotated_image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
                 
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image)                