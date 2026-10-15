import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads an image from path, resizes it to the target dimensions required by the CNN,
    converts it to a numpy array, normalizes pixel values to [0, 1], and adds a batch dimension.
    """
    # Load image with target size
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch of shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    img_array = img_array / 255.0
    
    return img_array