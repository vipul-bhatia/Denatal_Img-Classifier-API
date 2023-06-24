import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Loading a trained model
def load_model(model_path):
    print('Loading saved model')
    model = tf.keras.models.load_model(model_path,
                                      custom_objects={'KerasLayer': hub.KerasLayer})
    return model

model_path = f"{BASE_DIR}/abc-{__version__}.h5"
model = load_model(model_path)


classes = ['Gingivitis', 'Mouth_Ulcer', 'Calculus', 'Tooth_discoloration', 'Caries', 'hypodontia']

def predict(image):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = (image/255.0)
    image = tf.expand_dims(image, axis=0)
    pred = model.predict(image)
    pred_class = classes[pred.argmax()]
    return pred_class