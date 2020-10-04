from keras.applications import MobileNetV2, ResNet50
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Conv2D, Add, Activation, MaxPool2D
from keras.models import Model


def LargeInputMobileNetv2(large_in, num_classes=4):
    #skipcon = Conv2D(3,5,strides=(4,2), activation='relu', padding='same')(large_in)
    #x = Conv2D(6,3,strides=(2,2), activation='relu', padding='same')(skipcon)
    #x = Conv2D(9,3,strides=(2,2), activation='relu', padding='same')(x)
    #skipcon = Conv2D(9,5,strides=(4,4), activation='relu', padding='same')(skipcon)

    #x = Add()([x,skipcon])
    #x = Activation('relu')(x)


    skipcon = MaxPool2D(pool_size=(4,2), padding='same')(large_in)
    x = Conv2D(6,3,strides=(2,2), activation='relu', padding='same')(skipcon)
    x = Conv2D(9,3,strides=(2,2), activation='relu', padding='same')(x)
    skipcon = Conv2D(9,5,strides=(4,4), activation='relu', padding='same')(skipcon)

    x = Add()([x,skipcon])
    x = Activation('relu')(x)


    #model = MobileNetV2(input_tensor=x, include_top=False, pooling='max', weights=None)
    model = ResNet50(input_tensor=x, include_top=False, pooling='max', weights=None)
    features = model.output
    x = Dense(num_classes, activation='softmax')(features)

    tmodel = Model(large_in, x)

    return tmodel