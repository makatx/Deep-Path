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

def generateInput(slidefile, patch_predictions, folder='', level=5, dimensions=(6000, 3000)):
    '''
    file: file name of the slide
    folder: folder to look in
    dimensions: (height, width) of the output
    '''
    mask = buildMask(folder+slidefile, patch_predictions, level=5)
    mask = resizeMask(mask, dimensions)
    #mask = addNoise(mask)
    
    return mask.reshape((mask.shape[0], mask.shape[1],1))

def generateBatch(maskSource, folder, batch_size=10, level=5, dimensions=(6000,3000)):
    '''
    maskSource: dict object with slide filename as keys and [[coord], [probability]] as the value
    folder: where the slide files are located
    '''
    
    images = []
    y_train = []
    b = 0

    filelist = maskSource.keys()
    
    #print("File list: ", filelist)

    for file in filelist:
        if b==batch_size:
            b=0
            images_batch = np.array(images)
            images = []
            yield images_batch
        images.append(generateInput(file, patch_predictions=maskSource[file], folder, level=level, dimensions=dimensions))
        b+=1
    images_batch = np.array(images)
    images = []
    yield images_batch


if __name__ == '__main__':

    archs = ['inceptionv3', 'inceptionresnetv2', 'mobilenetv2']

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser(description='Train (or continue to train) large input CNN (mobilenet bb) model on dummy prediction score masks')
    aparser.add_argument('--predict-filelist', type=str, help='Specify the dictionary containing patch coordinates/predicitons per slide filename (as key)')
    aparser.add_argument('--batch-size', type=int, default=10, help='batch_size to use')
    aparser.add_argument('--log-dir', type=str, default='logs/', help='location to store fit.log (appended)')
    aparser.add_argument('--saves-name', type=str, default='_slide_labels_out.json', help='string to save output to (saved "incrementally" if file exists)')
    aparser.add_argument('--predict-level', type=int, default=5, help='the slide level to build masks from, and run through the network')
    aparser.add_argument('--slides-folder', type=str, default='', help='Path of the slides and annotation folder. Should be empty (or not set) if patch coord list (json) already has this')
    aparser.add_argument('--load-model-file', type=str, help='full path of the model save file to load')

    args = aparser.parse_args()

    date = str(datetime.now().date())
    dims = (6000, 3000)

    with open(args.predict_filelist, 'r') as f:
        predict_filelist = json.load(f)
    
    #in_generator = generateBatch(predict_filelist, folder=args.slides_folder, batch_size=args.batch_size, level=args.predict_level, dimensions=dims)
    #prediction_steps_per_epoch = math.ceil(len(predict_filelist)/args.batch_size)

    if args.load_model_file != None:
        model = load_model(args.load_model_file)
    else:
        print("Model file needed to continue. Exiting.")
        sys.exit(1)
   
    if os.path.exists(args.saves_name):
        with open(args.saves_name, 'r') as f:
            labels_predictions = json.load(f)
    else:
        labels_predictions = {}
    completed_slides = labels_predictions.keys()

    for slidefile in predict_filelist.keys():
        if slidefile in completed_slides:
            print("Skipping file: ", slidefile)
            continue
        in_image = generateInput(slidefile, predict_list[slidefile], folder=args.slides_folder, level=args.predict_level, dimensions=dims)
        prediction = model.predict([in_image], verbose=1)
        labels_predictions[slidefile] = prediction[0]

        #incrementally saving to hedge against interruptions and allowing resume-ability
        with open(args.saves_name, 'w') as f:
            json.dump(labels_predictions, f)
