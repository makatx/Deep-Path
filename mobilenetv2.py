import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Add, Input, Activation, Dense, Softmax, Flatten, Dropout
from keras.models import Model
from keras.activations import relu
from keras import backend as K
from keras import regularizers

### Define Conv2d with Regularization
l2 = 0.01
def Conv2D_l2(channels, kernel, padding='valid', strides=1, name='', l2=l2):
    return Conv2D(channels, kernel, padding=padding, strides=strides, name=name, kernel_regularizer=regularizers.l2(l2))
def Dense_l2(units, activation=None, l2=l2):
    return Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2))
def DepthwiseConv2D_l2(kernel_size, strides=1, padding='same', name='', l2=l2):
    return DepthwiseConv2D(kernel_size, strides=strides, padding=padding, name=name, depthwise_regularizer=regularizers.l2(l2))
### Define bottleneck blocks ###

def bottleneck_block_s1(x, end_points, layer_count, expansion_rate=6, depth_multiplier=1.0):
    in_channels = K.int_shape(x)[-1]
    expanded_channels = int(expansion_rate * in_channels * depth_multiplier)

    ## Expand ##
    endpoint = 'layer_%d_expansion_output'%layer_count
    m = Conv2D_l2(expanded_channels, (1,1), name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## DepthWiseConv ##
    endpoint = 'layer_%d_depthwise_output'%layer_count
    m = DepthwiseConv2D_l2(3, padding='same' , name=endpoint)(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## Project ##
    endpoint = 'layer_%d_projection_output'%layer_count
    m = Conv2D_l2(in_channels, (1,1), name=endpoint)(m)
    m = BatchNormalization()(m)
    end_points[endpoint] = m

    return Add()([m,x])

def bottleneck_block_s2(x, out_channels, end_points, layer_count, expansion_rate=6, stride=2, depth_multiplier=1.0):
    in_channels = K.int_shape(x)[-1]
    expanded_channels = int(expansion_rate * in_channels * depth_multiplier)
    out_channels = int(out_channels * depth_multiplier)
    #print('in_channels: {} | expanded_channels: {}'.format(in_channels, expanded_channels))

    ## Expand ##
    endpoint = 'layer_%d_expansion_output'%layer_count
    m = Conv2D_l2(expanded_channels, (1,1), name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## DepthWiseConv ##
    endpoint = 'layer_%d_depthwise_output'%layer_count
    m = DepthwiseConv2D_l2(3, padding='same', strides=stride, name=endpoint)(m)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m

    ## Project and change original channel count ##
    endpoint = 'layer_%d_projection_output'%layer_count
    m = Conv2D_l2(out_channels, (1,1), name=endpoint)(m)
    m = BatchNormalization()(m)
    end_points[endpoint] = m

    return m

### Define a sequence of Bottleneck blocks ###
def bottleneck_sequence(x, end_points, layer_count, out_channels='same', stride=1, expansion_rate=6, count=1, depth_multiplier=1.0):
    block_count = count

    m = x
    if out_channels != 'same' :
        m = bottleneck_block_s2(x, out_channels, end_points, layer_count, expansion_rate=expansion_rate, stride=stride, depth_multiplier=depth_multiplier)
        block_count -= 1
        layer_count +=1

    while block_count > 0 :
        m = bottleneck_block_s1(m, end_points, layer_count, expansion_rate=expansion_rate, depth_multiplier=depth_multiplier)
        block_count -= 1
        layer_count += 1

    return m

### Define the MobileNetV2 Feature Extractor model ###

def MobileNetV2_FE(x, output_stride=16, depth_multiplier=1.0):
    '''
    MobileNetV2 Feature Extractor
    Returns the output of the last 'Bottleneck' sequence
    Additionally returns early image-level features if return_IF is True

    Supports only output_stride=16 or 8
    '''
    end_points = {}
    layer_count = 1

    endpoint = 'layer_%d' % layer_count
    m = Conv2D_l2(32, (3,3), padding='same', strides=2, name=endpoint)(x)
    m = BatchNormalization()(m)
    m = Activation('relu')(m)
    end_points[endpoint] = m
    layer_count += 1

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=16, expansion_rate=1, depth_multiplier=depth_multiplier)
    layer_count += 1

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=24, stride=2, count=2, depth_multiplier=depth_multiplier)
    layer_count += 2

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=32, stride=2, count=3, depth_multiplier=depth_multiplier)
    layer_count += 3

    if output_stride == 8:
        stride = 1
    else:
        stride = 2
    m = bottleneck_sequence(m, end_points, layer_count, out_channels=64, stride=stride, count=4, depth_multiplier=depth_multiplier)
    layer_count += 4


    m = bottleneck_sequence(m, end_points, layer_count, out_channels=96, stride=1, count=3, depth_multiplier=depth_multiplier)
    layer_count += 3

    if output_stride == 16:
        stride = 1
    ## Making Stride=1 for the next block since we want a maximum of 16
    ## as output_stride

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=160, stride=stride, count=3, depth_multiplier=depth_multiplier)
    layer_count += 3

    m = bottleneck_sequence(m, end_points, layer_count, out_channels=320, stride=1, count=1, depth_multiplier=depth_multiplier)

    #m = Conv2D(1280, (1,1))(m)
    #m = BatchNormalization()(m)

    return m, end_points

def MobileNetv2Classifier(images, num_classes=2, output_stride=16, depth_multiplier=1.0):
    '''
    Creates and returns classifier model using mobilenetv2 architecture

    '''
    features, _ = MobileNetV2_FE(images, output_stride=output_stride, depth_multiplier=depth_multiplier)
    features = Flatten()(features)
    features = Dense_l2(1280, activation='relu')(features)

    features = Dropout(0.1)(features)

    logits = Dense_l2(num_classes)(features)
    probabilities = Softmax()(logits)

    return probabilities


if __name__ == '__main__':
    print('Creating the model for testing with input shape (16,16,3)...')
    input_image = Input(shape=(16,16,3))
    mbnet_features, end_points = MobileNetV2_FE(input_image)
    model = Model(input_image, mbnet_features)
    n = np.random.rand(16,16,3)
    n = np.expand_dims(n, 0)
    t = model.predict(n)

    print('model.predict() output shape: {} \n Output:\n {}'.format(t.shape, t))
