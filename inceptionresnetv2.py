from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras import layers
import numpy as np

def MIA_InceptionResNetV2(input_image, num_classes=2):
    base_model = InceptionResNetV2(weights=None, include_top=False, input_tensor=input_image, pooling=None)
    features = base_model.output
    x = layers.Flatten()(features)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    probabilities = layers.Dense(num_classes, activation='sigmoid')(x)

    return probabilities

if __name__ == '__main__':
    print('Creating the model for testing with input shape (16,16,3)...')
    input_image = Input(shape=(256,256,3))
    probs = MIA_InceptionResNetV2(input_image)
    model = Model(input_image, probs)
    n = np.random.rand(256,256,3)
    n = np.expand_dims(n, 0)
    t = model.predict(n)

    print('model.predict() output shape: {} \n Output:\n {}'.format(t.shape, t))
