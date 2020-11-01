import numpy as np
from math import ceil
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Input, Dense, Softmax
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger

from mobilenetv2 import MobileNetv2Classifier
from inceptionresnetv2 import MIA_InceptionResNetV2
from inceptionV3 import MIA_InceptionV3

from Slide import Slide
import json
import argparse
import sys
import os
from datetime import datetime

def patch_batch_generator(slide, tile_list, batch_size=32, level=1, dims=(256,256)):
    images = []
    b = 0
    for coord in tile_list:
        if b==batch_size:
            b=0
            images_batch = np.array(images)
            images = []
            yield images_batch

        img = slide.getRegionFromSlide(level=level, start_coord=coord, dims=dims).astype(np.float)
        img = (img - 128)/128
        images.append(img)
        b +=1
    images_batch = np.array(images)
    yield images_batch




if __name__ == '__main__':

    archs = ['inceptionv3', 'inceptionresnetv2', 'mobilenetv2']

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser(description='Run specified model on given slides\' list and save output to JSON file (resume-able if interrupted)')
    aparser.add_argument('architecture', type=str, choices=archs, default='inceptionresnetv2', help='Specifies the CNN model architecture to use')
    aparser.add_argument('--load-weights', type=str, help='full path of the checkpoint/model weights file to load')
    aparser.add_argument('--load-model-file', type=str, help='full path of the model save file to load (takes precedence over loading weights)')
    aparser.add_argument('--batch-size', type=int, default=128, help='batch_size to use')
    aparser.add_argument('--slides-list', required=True, type=str, help='full path of json format list of slides to run model on')
    aparser.add_argument('--slides-folder', type=str, default='', help='Path of the slides folder. Should be empty (or not set) if patch coord list (json) already has this')
    aparser.add_argument('--out-folder', type=str, default='predict_out/', help='output folder. If this files under here exist, it is used to skip slides recorded already and remainder are added as processed and saved.  Expected file name is "slide_name.json"')
    aparser.add_argument('--run-level', choices=[0,1], type=int, default=1, help='the slide/zoom level the network should run on')
    aparser.add_argument('--patch-extraction-level', choices=[5,6,7], type=int, default=7, help='Level at which to run thresholding (specified by --thresh-method to extract pixels of informational areas on slide. Coordinates of Pixels is used as patch coordinates')
    aparser.add_argument('--thresh-method', choices=['None', 'OTSU','HED', 'GRAY'], type=str, default='GRAY', help='if \'None\', consecutive tiles are extracted from the image at --train-level with specified --overlap, otherwise specidifed method is used to pick out useful pixels as coordiantes at given extraction_level' )
    aparser.add_argument('--overlap', choices=[0,0.25,0.5,0.75], type=float, default=0.25, help='Overlap between consecutive patches (used if thresh_method is None)')
    aparser.add_argument('--patch-size', choices=[256], type=int, default=256, help='patch_size')
    aparser.add_argument('--rect', type=str, default='0', help='area to run the model prediction on.\nCoordinates are from level-0. \nProvide list as a string, ex: [[0,0][27489,36985]]')

    args = aparser.parse_args()

    with open(args.slides_list, 'r') as f:
        slides_list = json.load(f)
    area = json.loads(args.rect)
    dims = (args.patch_size, args.patch_size)
    if args.thresh_method == 'None': 
        thresh_method=None
    else:
        thresh_method = args.thresh_method
    
    if args.load_model_file != None:
        model = load_model(args.load_model_file)
    else:
        input_patch = Input(shape=(dims[0],dims[1],3,))
        if args.architecture == 'mobilenetv2':
            depth_multiplier = 0.75
            probs = MobileNetv2Classifier(input_patch, num_classes=2, output_stride=32, depth_multiplier=depth_multiplier)
        elif args.architecture == 'inceptionv3':
            probs = MIA_InceptionV3(input_patch, num_classes=2)
        elif args.architecture == 'inceptionresnetv2':
            probs = MIA_InceptionResNetV2(input_patch, num_out=2)

        model = Model(input_patch, probs)

    if args.load_weights != None:
        model.load_weights(args.load_weights)
        print('Loaded weights from {}'.format(args.load_weights))

    ls = os.listdir(args.out_folder)
    completed_slides = []
    for file in ls:
        completed_slides.append(os.path.splitext(file)[0])

    complete_count = len(completed_slides)

    for slidename in slides_list:
        if slidename in completed_slides: 
            print('Skipping file: ', slidename)
            continue

        slide = Slide(args.slides_folder+slidename)
        tile_list = slide.getTileList(thresh_method=thresh_method, view_level=args.run_level, extraction_level=args.patch_extraction_level, area=area, patch_size=dims[0], overlap=args.overlap)
        gen = patch_batch_generator(slide, tile_list, batch_size=args.batch_size, level=args.run_level, dims=dims)

        print("\nCompleted {} of {}".format(complete_count, len(slides_list)))
        print("Running on:", slidename)
        predictions = model.predict_generator(gen, ceil(len(tile_list)/args.batch_size), verbose=1)

        save_name = slidename + '.json'
        with open(args.out_folder+save_name, 'w') as f:
            json.dump({slidename:list(zip(tile_list, predictions.tolist()))}, f)
        complete_count += 1



