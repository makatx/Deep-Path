from keras.applications import MobileNetV2
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Conv2D
from keras.models import Model


def LargeInputMobileNetv2(large_in, num_classes=4):
    x = Conv2D(3,50,strides=(8,4), activation='relu', padding='same')(large_in)
    x = Conv2D(3,25,strides=(2,2), activation='relu')(x)
    model = MobileNetV2(input_tensor=x, include_top=False, pooling='max', weights=None)
    features = model.output
    x = Dense(4, activation='softmax')(features)

    tmodel = Model(large_in, x)

    return tmodel