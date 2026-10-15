import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

def make_gradcam_heatmap(img_path, model, last_conv_layer_name="conv5_block3_out"):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="static/heatmaps/heatmap.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    
    # Fix: use applyColorMap instead of colormap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    superimposed_img = jet * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path