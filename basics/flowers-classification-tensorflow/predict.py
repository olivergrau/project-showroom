import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import argparse
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

def process_image(image_path):
    """Scales, crops, and normalizes a PIL image,
       returns a TensorFlow tensor
    """
    image = Image.open(image_path)

    # Scale the image to shortest side and keep aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        new_size = (int(aspect_ratio * 256), 256)
    else:
        new_size = (256, int(256 / aspect_ratio))

    image = image.resize(new_size)

    # Crop around the center
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224

    image = image.crop((left, top, right, bottom))

    # Convert to numpy array
    np_image = np.array(image)

    # Normalize to 0-1
    np_image = np_image / 255.0

    # TensorFlow expects images in the format (height, width, channels)
    # So no need to transpose like PyTorch

    # Convert to TensorFlow tensor
    tf_image = tf.convert_to_tensor(np_image, dtype=tf.float32)

    return tf_image

def predict(image_path, model, class_names, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    # process image
    loaded_img = process_image(image_path)

    # extend image with a batch dimension, so the net can work with it
    loaded_img = tf.expand_dims(loaded_img, axis=0)

    # model must be in the prediction mode
    model.trainable = False

    output = model(loaded_img)

    # top-k probabilities
    top_probs, top_indices = tf.nn.top_k(output, k=topk)

    # we use the named labes from the orginal oxford flower set
    classes = [class_names[str(idx)] for idx in top_indices[0].numpy()]

    return top_probs[0].numpy(), classes

def main():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")

        return ivalue
    
    parser = argparse.ArgumentParser(description="Predict class of a provided image")

    # basic usage    
    parser.add_argument('image_path', type=str, help='Path to the image which should be predicted')
    parser.add_argument('saved_model', type=str, help='Path to the saved model info for pretrained net')

    # options        
    parser.add_argument('--top_k', type=positive_int, help='Top k classes which are predicted', default=5)
    parser.add_argument('--category_names', type=str, help='Category mapping', default=None)

    # parse args
    args = parser.parse_args()

    # print args and options
    print(f"Path to image: {args.image_path}")
    print(f"Path to saved model file: {args.saved_model}")
    print(f"Top k classes: {args.top_k}")
    print(f"Path to category file: {args.category_names}")
    
    if args.category_names is None:
        with open('label_map.json', 'r') as f:
            class_names = json.load(f)
    else:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)

    # Load the model
    model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Predict (Predicting Classes)
    probs, classes = predict(args.image_path, model, class_names, args.top_k)

    # Print results (Top K Classes and Displaying Class Names)
    print(f"Top {args.top_k} class directories: {classes}")
    
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
