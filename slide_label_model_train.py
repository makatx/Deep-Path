import numpy as np
import math
import cv2, os
from Slide import Slide
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dense, Softmax
from keras import backend as K
from keras.optimizers import SGD, Adam, Adadelta, Nadam
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from LargeInMB import LargeInputMobileNetv2

import json
import argparse
import sys
from datetime import datetime

LEARNING_RATE = 0.001
DECAY_RATE = 0.99
DECAY_STEP = 10

'''
Get the mask of slide and resize it, followed by gaussian noising. Provide back this mask
'''

def getMask(file, level):
    slide = Slide(file, annot_file=os.path.splitext(file)[0]+'.xml')
    mask = slide.getGTmask((0,0), dims='full', level=level)
    return mask.reshape((mask.shape[0], mask.shape[1]))

def buildMask(file, patch_predictions, level=5, prediction_level = 1, patch_dim=256):
    '''
    Given the list of patch coordinates and their model predictied probablities in patch_predictions (ex of single list item: [[12312,89709],[0.2,0.8]]),
    and the dimensions of the patch, build the prediction mask at given level for the slide from given file 
    Tile areas that overlap are averaged 
    '''
    #print("File name from buildMask: ", file)
    slide = Slide(file)
    patch_scale = slide.slide.level_downsamples[prediction_level]/slide.slide.level_downsamples[level]
    scaled_dim = int(patch_dim * patch_scale)

    coord_scale = slide.slide.level_downsamples[level]

    mask = np.zeros(slide.slide.level_dimensions[level][::-1])
    mask_counter = np.zeros_like(mask)
    for coord_l0, score in patch_predictions:
        coord = [int(coord_l0[0]/coord_scale), int(coord_l0[1]/coord_scale)]

        mask[coord[1]:coord[1]+scaled_dim, coord[0]:coord[0]+scaled_dim] += score[1]
        mask_counter[coord[1]:coord[1]+scaled_dim, coord[0]:coord[0]+scaled_dim] += 1

    mask_counter[mask_counter==0] =1

    return mask/mask_counter

def resizeMask(mask, dimensions):
    if dimensions[0]>dimensions[1] and mask.shape[1]>mask.shape[0]:
        mask = np.transpose(mask)
        assert mask.shape[0]>=mask.shape[1]
    elif dimensions[0]<dimensions[1] and mask.shape[1]<mask.shape[0]:
        mask = np.transpose(mask)
        assert mask.shape[0]<=mask.shape[1]
        
    mask = cv2.resize(mask, dimensions[::-1])
    return mask

def addNoise(img, mu=0, sigma=0.001):
    noise = np.random.normal(mu,sigma, img.shape)
    noisy_mask = img+noise
    noisy_mask[noisy_mask>1]=1
    noisy_mask[noisy_mask<0]=0
    gaus_noisy_mask = cv2.GaussianBlur(noisy_mask, (55,55), 1)
    
    return gaus_noisy_mask

def generateTrainingInput(slidefile, folder='', level=5, dimensions=(6000, 3000), patch_predictions=[]):
    '''
    file: file name of the slide
    folder: folder to look in
    dimensions: (height, width) of the output
    '''
    if len(patch_predictions)==0:
        mask = getMask(folder+slidefile, level)
    else:
        mask = buildMask(folder+slidefile, patch_predictions, level=5)
    mask = resizeMask(mask, dimensions)
    mask = addNoise(mask)
    
    return mask.reshape((mask.shape[0], mask.shape[1],1))

def generateBatch(maskSource, gt, folder, batch_size=10, level=5, dimensions=(6000,3000)):
    images = []
    y_train = []
    b = 0

    if type(maskSource)==type([]):
        filelist = maskSource
        gen=True
    elif type(maskSource)==type({}):
        filelist = maskSource.keys()
        gen=False

    #print("File list: ", filelist)

    for file in filelist:
        if b==batch_size:
            b=0
            images_batch = np.array(images)
            y_train_batch = np.array(y_train)
            images, y_train = [], []
            yield images_batch, y_train_batch
        if gen:
            images.append(generateTrainingInput(file, folder, level=level, dimensions=dimensions))
        else:
            images.append(generateTrainingInput(file, folder, level=level, dimensions=dimensions, patch_predictions=maskSource[file]))
        label = [0,0,0,0]
        label[gt[file]]=1
        y_train.append(label)
        b+=1
    images_batch = np.array(images)
    y_train_batch = np.array(y_train)
    images, y_train = [], []
    yield images_batch, y_train_batch


if __name__ == '__main__':

    archs = ['inceptionv3', 'inceptionresnetv2', 'mobilenetv2']

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser(description='Train (or continue to train) large input CNN (mobilenet bb) model on dummy prediction score masks')
    aparser.add_argument('--train-filelist', type=str, help='Specify the file list for training set or the dictionary containing patch coordinates/predicitons per slide filename (as key)')
    aparser.add_argument('--test-filelist', type=str, help='Specify the file list for test set')
    aparser.add_argument('--gt-file', type=str, help='Specify the ground truth file')
    aparser.add_argument('--batch-size', type=int, default=10, help='batch_size to use')
    aparser.add_argument('--initial-epoch', type=int, default=0, help='starting epoch number to use for this run')
    aparser.add_argument('--epochs', type=int, default=1, help='number of epochs to run the training for')
    aparser.add_argument('--learning-rate', type=float, default=1e-2, help='learning rate')
    aparser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='location to store checkpoints/model weights after each epoch')
    aparser.add_argument('--log-dir', type=str, default='logs/', help='location to store fit.log (appended)')
    aparser.add_argument('--saves-name', type=str, default='__', help='string to add in checkpoints and model saves file names')
    aparser.add_argument('--optimizer', choices=['adadelta', 'adam', 'sgd', 'nadam'], type=str, default='adam', help='Optimizer to use (default parameters)')
    aparser.add_argument('--nesterov', type=bool, default=False, help='Use Nesterov with SGD?')
    aparser.add_argument('--momentum', type=float, default=0, help='momentum with SGD')
    aparser.add_argument('--decay-lr', type=bool, default=False, help='Optimizer to use (default parameters)')
    aparser.add_argument('--train-level', type=int, default=5, help='the slide levelto generate masks from')
    aparser.add_argument('--slides-folder', type=str, default='', help='Path of the slides and annotation folder. Should be empty (or not set) if patch coord list (json) already has this')
    aparser.add_argument('--load-model-file', type=str, help='full path of the model save file to load')

    args = aparser.parse_args()

    date = str(datetime.now().date())
    dims = (6000, 3000)

    with open(args.train_filelist, 'r') as f:
        train_filelist = json.load(f)
    with open(args.test_filelist, 'r') as f:
        test_filelist = json.load(f)
    with open(args.gt_file, 'r') as f:
        gt_dict = json.load(f)

    train_generator = generateBatch(train_filelist, gt_dict, folder=args.slides_folder, batch_size=args.batch_size, level=args.train_level, dimensions=dims)
    validn_generator = generateBatch(test_filelist, gt_dict, folder=args.slides_folder, batch_size=args.batch_size, level=args.train_level, dimensions=dims)

    train_steps_per_epoch = math.ceil(len(train_filelist)/args.batch_size)
    validn_steps_per_epoch = math.ceil(len(test_filelist)/args.batch_size)

    if args.load_model_file != None:
        model = load_model(args.load_model_file)
    else:
        mask_in = Input(shape=(dims[0],dims[1],1,))
        model = LargeInputMobileNetv2(mask_in, num_classes=4)
    
    checkpointer = ModelCheckpoint(args.checkpoint_dir+date+'_slideLabelCNN' +'_'+str(args.learning_rate)+'_weights_'+args.saves_name+'_{epoch:02d}--{categorical_accuracy:.4f}--{val_loss:.4f}.hdf5', monitor='categorical_accuracy',
                            save_weights_only=True, save_best_only=True)
    csvlogger = CSVLogger(args.log_dir+'_'+date+'_slideLabelCNN_'+args.saves_name+'_fit.log', append=True)
    callbacks = [checkpointer, csvlogger]

    if args.optimizer=='sgd':
        opt = SGD(lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimizer=='adam':
        opt = Adam(lr=args.learning_rate)
    elif args.optimizer=='adadelta':
        opt = Adadelta()
    elif args.optimizer=='nadam':
        opt = Nadam()
    else:
        opt = args.optimizer
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=args.epochs, verbose=1, callbacks=callbacks,
    validation_data=validn_generator, validation_steps=validn_steps_per_epoch, initial_epoch=args.initial_epoch)

    last_epoch = args.epochs
    model.save('modelsaves/'+ date + '_slideLabelCNN_'+args.saves_name+'_afterEpoch-'+str(last_epoch)+'.h5')
