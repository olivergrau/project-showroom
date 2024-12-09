import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import random
from torchvision import transforms
import json

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])

        # Fuck, this already scales down the images between 0 and 1. I wasted nearly 1 weeks with this...
        # --> image = mpimg.imread(image_name) # damn fucking auto-scaling functionality of matplotlib <-- culprit
        image = cv2.imread(image_name)  # Loads in BGR format by default in range 0-255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # If image has an alpha channel, remove it
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Keypoints extraction
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts, 'name': self.key_pts_frame.iloc[idx, 0]}
    
        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)    
    
        return sample

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __init__(self, mean=None, std=None):
        self.mean = mean  # of the training dataset
        self.std = std    # of the training dataset

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        
        # Step 1: Convert image to grayscale
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        
        # Step 2: Scale color range from [0, 255] to [0, 1]
        image_copy = image_copy / 255.0
        
        # Step 3: Apply z-score standardization
        # (image - dataset_mean) / dataset_std.
        if self.mean is not None and self.std is not None:
            image_copy = (image_copy - self.mean) / self.std
        
        # Step 4: Normalize keypoints to be centered around 0
        key_pts_copy = (key_pts_copy - 100) / 50.0
        
        return {'image': image_copy, 'keypoints': key_pts_copy, 'name': sample['name']}

class RandomRescale:
    def __init__(self, scale_range=(0.8, 1.2), output_size=(224, 224)):
        self.scale_range = scale_range
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # Choose random scale
        scale_factor = random.uniform(*self.scale_range)

        # Compute new dimensions
        new_h, new_w = int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)

        # Resize image
        scaled_image = cv2.resize(image, (new_w, new_h))

        # Scale keypoints accordingly
        scaled_keypoints = keypoints * scale_factor

        # Center crop/pad to output size
        scaled_image = cv2.resize(scaled_image, self.output_size)
        scale_x, scale_y = self.output_size[0] / new_w, self.output_size[1] / new_h
        scaled_keypoints = scaled_keypoints * [scale_x, scale_y]

        return {'image': scaled_image, 'keypoints': scaled_keypoints, 'name': sample['name']}


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

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
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts, 'name': sample['name']}



class Sharpen(object):
    """Apply sharpening to the image in a sample using either a basic kernel or unsharp masking.
    
    Args:
        method (str): Sharpening method to use: 'basic' for a sharpening kernel or 'unsharp' for unsharp masking.
        strength (float): Strength of sharpening. Only used in unsharp masking.
        blur_size (tuple): Kernel size for Gaussian blur in unsharp masking. Only used in unsharp masking.
    """

    def __init__(self, method='unsharp', strength=1.5, blur_size=(5, 5)):
        assert method in ['basic', 'unsharp'], "method must be 'basic' or 'unsharp'"
        self.method = method
        self.strength = strength
        self.blur_size = blur_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        if self.method == 'basic':
            # Define a basic sharpening kernel
            sharpening_kernel = np.array([[0, -1, 0],
                                          [-1, 5, -1],
                                          [0, -1, 0]])
            # Apply the kernel to the image
            sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

        elif self.method == 'unsharp':
            # Apply unsharp masking
            blurred = cv2.GaussianBlur(image, self.blur_size, sigmaX=0)
            sharpened_image = cv2.addWeighted(image, 1 + self.strength, blurred, -self.strength, 0)

        return {'image': sharpened_image, 'keypoints': key_pts, 'name': sample['name']}

class GaussianBlur(object):
    """Apply Gaussian Blur to the image in a sample.

    Args:
        kernel_size (int or tuple): Size of the Gaussian kernel.
        sigma (float): Standard deviation for Gaussian kernel.
    """

    def __init__(self, kernel_size=(3, 3), sigma=0.1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Apply Gaussian blur to the image
        img = cv2.GaussianBlur(image, self.kernel_size, self.sigma)

        return {'image': img, 'keypoints': key_pts, 'name': sample['name']}

import numpy as np

import numpy as np

class GrayscaleJitter(object):
    """Randomly change the brightness and contrast of a grayscale image in a sample.
    
    Args:
        brightness_delta (float): Maximum delta to adjust brightness (in range [0, 1]).
                                  0.1 means brightness changes up to ±10%.
        contrast_delta (float): Maximum delta to adjust contrast (in range [0, 1]).
                                0.1 means contrast changes up to ±10%.
    """

    def __init__(self, brightness_delta=0.1, contrast_delta=0.1):
        # Define max deltas for small adjustments
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Randomize brightness and contrast factors within controlled delta range
        brightness_factor = 1.0 + np.random.uniform(-self.brightness_delta, self.brightness_delta)
        contrast_factor = 1.0 + np.random.uniform(-self.contrast_delta, self.contrast_delta)

        # Adjust contrast: normalize around mean to avoid over-darkening or over-lightening
        mean_intensity = np.mean(image)
        img = (image - mean_intensity) * contrast_factor + mean_intensity

        # Adjust brightness with a subtle factor
        img = img + (brightness_factor - 1) * 127  # Adjust around midpoint for grayscale

        # Clip values to valid range [0, 255] to avoid saturation
        img = np.clip(img, 0, 255).astype(np.uint8)

        return {'image': img, 'keypoints': key_pts, 'name': sample['name']}


class RandomHorizontalFlip(object):
    """Horizontally flip the image in a sample with a given probability.

    Args:
        prob (float): Probability to apply the horizontal flip (0.0 to 1.0).
    """

    def __init__(self, prob=0.25):
        self.apply_probability = prob

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Apply flip based on probability
        if random.random() > self.apply_probability:
            return sample

        # Flip the image horizontally
        flipped_image = cv2.flip(image, 1)

        # Flip keypoints horizontally
        h, w = image.shape[:2]
        flipped_key_pts = key_pts.copy()
        flipped_key_pts[:, 0] = w - key_pts[:, 0]  # Reflect x-coordinates across the center

        return {'image': flipped_image, 'keypoints': flipped_key_pts, 'name': sample['name']}

class FaceCrop(object):
    def __init__(self, padding=10, output_size=(224, 224)):
        self.padding = padding
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Calculate bounding box around keypoints
        x_min, y_min = np.min(key_pts, axis=0)
        x_max, y_max = np.max(key_pts, axis=0)

        # Add padding
        x_min = max(0, int(x_min - self.padding))
        y_min = max(0, int(y_min - self.padding))
        x_max = min(image.shape[1], int(x_max + self.padding))
        y_max = min(image.shape[0], int(y_max + self.padding))

        # Crop the image
        face_crop = image[y_min:y_max, x_min:x_max]

        # Resize to the desired output size
        face_crop_resized = cv2.resize(face_crop, self.output_size)

        # Scale keypoints to the resized cropped image
        scale_x = self.output_size[0] / (x_max - x_min)
        scale_y = self.output_size[1] / (y_max - y_min)
        key_pts_rescaled = (key_pts - [x_min, y_min]) * [scale_x, scale_y]

        return {'image': face_crop_resized, 'keypoints': key_pts_rescaled, 'name': sample['name']}

class RandomFaceCropAndResize(object):
    def __init__(self, padding=10, output_size=(224, 224), prob=0.5):
        self.padding = padding
        self.output_size = output_size
        self.probability = prob

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Apply crop based on probability
        if np.random.rand() > self.probability:
            # Resize image without cropping and scale keypoints
            image_resized = cv2.resize(image, self.output_size)
            scale_x = self.output_size[0] / image.shape[1]
            scale_y = self.output_size[1] / image.shape[0]
            key_pts_rescaled = key_pts * [scale_x, scale_y]
            return {'image': image_resized, 'keypoints': key_pts_rescaled, 'name': sample['name']}

        # Calculate bounding box around keypoints
        x_min, y_min = np.min(key_pts, axis=0)
        x_max, y_max = np.max(key_pts, axis=0)

        # Add padding
        x_min = max(0, int(x_min - self.padding))
        y_min = max(0, int(y_min - self.padding))
        x_max = min(image.shape[1], int(x_max + self.padding))
        y_max = min(image.shape[0], int(y_max + self.padding))

        # Crop the image
        face_crop = image[y_min:y_max, x_min:x_max]

        # Resize to the desired output size
        face_crop_resized = cv2.resize(face_crop, self.output_size)

        # Scale keypoints to the resized cropped image
        scale_x = self.output_size[0] / (x_max - x_min)
        scale_y = self.output_size[1] / (y_max - y_min)
        key_pts_rescaled = (key_pts - [x_min, y_min]) * [scale_x, scale_y]

        return {'image': face_crop_resized, 'keypoints': key_pts_rescaled, 'name': sample['name']}

class RandomRotate(object):
    """Rotate the image in a sample by a given angle in a specified percentage of cases.

    Args:
        angle (int): Rotation angle in degrees.
        prob (float): Probability to apply the rotation (0.0 to 1.0).
    """

    def __init__(self, angle=(-10, 10), prob=0.5):
        self.angle = angle
        self.apply_probability = prob

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Randomly decide whether to apply rotation based on the apply_probability
        if random.random() > self.apply_probability:
            return sample

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # randomly get an angle between 0 and self.angle
        angle = random.uniform(self.angle[0], self.angle[1])
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Rotate the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Rotate keypoints
        ones = np.ones((key_pts.shape[0], 1))
        keypoints_with_ones = np.hstack([key_pts, ones])
        rotated_keypoints = rotation_matrix.dot(keypoints_with_ones.T).T

        return {'image': rotated_image, 'keypoints': rotated_keypoints, 'name': sample['name']}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # Ensure the crop size does not exceed the image size
        if h < new_h or w < new_w:
            # If the image is too small, resize it to the desired crop size
            image = cv2.resize(image, (new_w, new_h))
            key_pts = key_pts * [new_w / w, new_h / h]  # Adjust keypoints accordingly
            top, left = 0, 0  # Crop from the top-left corner

        else:
            # Randomly select a top-left point for cropping
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            # Perform the crop
            image = image[top: top + new_h, left: left + new_w]
            # Adjust keypoints
            key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts, 'name': sample['name']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float(), 'name': sample['name']}