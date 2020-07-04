import numpy as np
import math
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dense, Softmax
from keras import backend as K
from keras.optimizers import SGD, Adam, Adadelta, Nadam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from PatchGenerator import patch_generator
from mobilenetv2 import MobileNetv2Classifier
from inceptionresnetv2 import MIA_InceptionResNetV2
from inceptionV3 import MIA_InceptionV3


import json
import argparse
import sys
from datetime import datetime

LEARNING_RATE = 0.001
DECAY_RATE = 0.99
DECAY_STEP = 10

def lr_schedule(steps):
    return LEARNING_RATE * DECAY_RATE ** (steps//DECAY_STEP)

if __name__ == '__main__':

    archs = ['inceptionv3', 'inceptionresnetv2', 'mobilenetv2']

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser(description='Train (or continue to train) specified model on patches')
    aparser.add_argument('architecture', type=str, choices=archs, default='inceptionresnetv2', help='Specifies the CNN model architecture to train')
    aparser.add_argument('--load-weights', type=str, help='full path of the checkpoint/model weights file to load')
    aparser.add_argument('--load-model-file', type=str, help='full path of the model save file to load')
    aparser.add_argument('--batch-size', type=int, default=32, help='batch_size to use')
    aparser.add_argument('--initial-epoch', type=int, default=0, help='starting epoch number to use for this run')
    aparser.add_argument('--epochs', type=int, default=1, help='number of epochs to run the training for')
    aparser.add_argument('--learning-rate', type=float, default=1e-2, help='learning rate')
    aparser.add_argument('--sample-factor', type=float, default=1, help='the ratio of true sample to negative samples')
    aparser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='location to store checkpoints/model weights after each epoch')
    aparser.add_argument('--log-dir', type=str, default='logs/', help='location to store fit.log (appended)')
    aparser.add_argument('--patch-list', required=True, type=str, help='full path of all_patch_list json file to load the patches lists dictionary (keys: negative, annotation, neighbor)')
    aparser.add_argument('--saves-name', type=str, default='__', help='string to add in checkpoints and model saves file names')
    aparser.add_argument('--optimizer', choices=['adadelta', 'adam', 'sgd', 'nadam'], type=str, default='sgd', help='Optimizer to use (default parameters)')
    aparser.add_argument('--nesterov', type=bool, default=False, help='Use Nesterov with SGD?')
    aparser.add_argument('--momentum', type=float, default=0, help='momentum with SGD')
    aparser.add_argument('--decay-lr', type=bool, default=False, help='Optimizer to use (default parameters)')
    aparser.add_argument('--train-level', choices=[0,1], type=int, default=1, help='the slide/zoom level the network should train on')
    aparser.add_argument('--slides-folder', type=str, default='', help='Path of the slides folder. Should be empty (or not set) if patch coord list (json) already has this')

    args = aparser.parse_args()

    batch_size = args.batch_size
    load_weights = args.load_weights
    load_model_file = args.load_model_file
    initial_epoch = args.initial_epoch
    epochs = args.epochs
    learning_rate = args.learning_rate
    sample_factor = args.sample_factor
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    saves_name = args.saves_name
    train_levels = [args.train_level]

    LEARNING_RATE = learning_rate

    date = str(datetime.now().date())

    print("Received following parameters:")
    print("Architecture={} \n batch_size={} \n checkpoint_dir={} \n epochs={} \n initial_epoch={} \n learning_rate={} \
    \n load_weights={} \n log_dir={} \n sample_factor={}".format(args.architecture,batch_size, checkpoint_dir,
                                                                 epochs, initial_epoch, learning_rate,
                                                                 load_weights, log_dir, sample_factor))

    print('\nCheckpoint to be saved as:\n',
    checkpoint_dir+date+'_' + args.architecture +'_'+'_weights_'+saves_name+'_{epoch:02d}--{categorical_accuracy:.4f}--{val_loss:.4f}.hdf5')

    print('\nModel to be saved as:\n',
    'modelsaves/'+date+'_' + args.architecture +'_'+saves_name+'_afterEpoch-'+str(epochs)+'.h5')

    if args.nesterov:
        print("###################### NESTEROV TRUE ###########")
    print('Using following patch list json files: \t{}'.format(args.patch_list))

    with open(args.patch_list, 'rb') as f:
        all_patch_list = json.load(f)

    train_neg_list, test_neg_list = train_test_split(all_patch_list['negative'], test_size=0.1)
    train_true_list, test_true_list = train_test_split(all_patch_list['annotation'], test_size=0.1)
    train_neigh_list, test_neigh_list = train_test_split(all_patch_list['neighbor'], test_size=0.1)

    dims = (256,256)
    input_patch = Input(shape=(dims[0],dims[1],3,))

    if load_model_file != None:
        model = load_model(load_model_file)
    else:
        if args.architecture == 'mobilenetv2':
            depth_multiplier = 0.75
            probs = MobileNetv2Classifier(input_patch, num_classes=2, output_stride=32, depth_multiplier=depth_multiplier)
        elif args.architecture == 'inceptionv3':
            probs = MIA_InceptionV3(input_patch, num_classes=2)
        elif args.architecture == 'inceptionresnetv2':
            probs = MIA_InceptionResNetV2(input_patch, num_classes=2)

        model = Model(input_patch, probs)
    checkpointer = ModelCheckpoint(checkpoint_dir+date+'_' + args.architecture +'_'+str(learning_rate)+'_weights_'+saves_name+'_{epoch:02d}--{categorical_accuracy:.4f}--{val_loss:.4f}.hdf5', monitor='categorical_accuracy',
                            save_weights_only=True, save_best_only=True)
    csvlogger = CSVLogger(log_dir+'_'+date+'_' + args.architecture +'_'+saves_name+'_fit.log', append=True)
    callbacks = [checkpointer, csvlogger]
    if args.decay_lr:
        print("##################Decay Learning Rate True################")
        lrs = LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lrs)

    
    if args.optimizer=='sgd':
        opt = SGD(lr=learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimizer=='adam':
        opt = Adam(lr=learning_rate)
    elif args.optimizer=='adadelta':
        opt = Adadelta()
    elif args.optimizer=='nadam':
        opt = Nadam()
    else:
        opt = args.optimizer

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    train_generator = patch_generator(args.slides_folder,
                                train_neg_list, train_true_list, train_neigh_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=train_levels)
    sublist_size = math.ceil(len(train_true_list)/sample_factor) - len(train_neigh_list)
    sampleset_size_train = len(train_neg_list[:sublist_size]) + len(train_true_list) +len(train_neigh_list)
    steps_per_epoch = math.ceil(sampleset_size_train/batch_size)
    #print("model_train: sampleset_size_train: ", sampleset_size_train)

    validn_generator = patch_generator(args.slides_folder,
                                test_neg_list, test_true_list, test_neigh_list,
                                sample_factor=sample_factor,
                                batch_size=batch_size, dims=dims, levels=train_levels)

    sublist_size = math.ceil(len(test_true_list)/sample_factor) - len(test_neigh_list)
    sampleset_size_validn = len(test_neg_list[:sublist_size]) + len(test_true_list) + len(test_neigh_list)
    steps_per_epoch_validn = math.ceil(sampleset_size_validn/batch_size)


    if load_weights != None:
        model.load_weights(load_weights)
        print('Loaded weights from {}'.format(load_weights))


    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks,
    validation_data=validn_generator, validation_steps=steps_per_epoch_validn, initial_epoch=initial_epoch)

    last_epoch = epochs
    model.save('modelsaves/'+date+'_' + args.architecture +'_'+saves_name+'_afterEpoch-'+str(last_epoch)+'.h5')
