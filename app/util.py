import numpy as np
import tensorflow as tf

IMG_SIZE = 224
def process_image(image_data):
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image,channels=3)
    # image = tf.image.convert_image_dtype(image,tf.float32)
    # image = tf.image.resize(image , size = [IMG_SIZE,IMG_SIZE])
    
    # return image
    image = tf.image.decode_image(image_data, expand_animations = False)
    print(f"Image shape after decoding: {image.shape}") 
    if len(image.shape) < 3:
        raise ValueError("Failed to decode the image, could not determine its shape.")
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

unique_names = ['Calculus',
  'Caries',
  'Gingivitis',
  'Mouth_Ulcer',
  'Tooth_discoloration',
  'hypodontia']
def get_pred_label(prediction_probablities):
    return unique_names[np.argmax(prediction_probablities)]

BATCH_SIZE = 32
def create_data_batches(x,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
    if test_data:
        print('Creating Test Data')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch