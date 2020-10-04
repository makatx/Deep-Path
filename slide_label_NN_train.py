import json, os, sys, numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, average_precision_score
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

FEATURE_ORDER = [
    'number_of_clusters',
    'detection_total_area',
    'detection_largest_area',
    'major_axis_length_largest_cluster',
    'average_probability_overall',
    'average_probability_largest_cluster',
    'maximum_probabilty_overall',
    'maximum_probabilty_largest_cluster',
    'foreground_area',
    'detection_to_foreground_ratio',
    'slide_area'
]

def MLPerceptron(input_dims=11, output_dims=4, hidden_layers=90):
    '''
    Define and return multilayer perceptron
    '''
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dims,), activation='tanh'))
    for i in range(hidden_layers):
        model.add(Dense(64, activation='tanh'))
    model.add(Dense(output_dims, activation='softmax'))
    return model



def getXYset(featureset, labels, subset=None):
    '''
    Return X_train, y_train list
    params:
        featureset: dict with slide names as keys and feature dict as value
        lables: dict of slide_name:numberic_label 
        subset: list of slide_names to extract from the featureset and return XY for
    Feature ordering:
        'number_of_clusters',
        'detection_total_area',
        'detection_largest_area',
        'major_axis_length_largest_cluster',
        'average_probability_overall',
        'average_probability_largest_cluster',
        'maximum_probabilty_overall',
        'maximum_probabilty_largest_cluster',
        'foreground_area',
        'detection_to_foreground_ratio',
        'slide_area'
    '''
    slides_list = (subset if subset != None else featureset.keys())
    X_train, y_train = [], []

    for s in slides_list:
        if labels[s]==0 and np.random.rand()>=0.1:
            continue
        fs = featureset[s]
        x = []
        for feature in FEATURE_ORDER:
            x.append(fs[feature])
        X_train.append(x)
        l = [0,0,0,0]
        l[labels[s]] = 1
        y_train.append(l)
    
    return shuffle(X_train, y_train)

if __name__ == "__main__":
    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser(description='Train lNN model on features extracted from slides')
    aparser.add_argument('--featureset', type=str, help='Specify the featureset dict file')
    aparser.add_argument('--labels', type=str, help='Specify the labels dict')
    aparser.add_argument('--subset-train', type=str, default='', help='File containing training slides file list, used to create trainset from featureset/labels')
    aparser.add_argument('--subset-test', type=str, default='', help='File containing testing slides file list, used to create testset from featureset/labels')
    aparser.add_argument('--savename', type=str, help='output save file name')

    args = aparser.parse_args()

    date = str(datetime.now().date())

    with open(args.featureset, 'r') as f:
        featureset = json.load(f)

    with open(args.labels, 'r') as f:
        labels = json.load(f)

    with open(args.subset_train, 'r') as f:
        subset_train = json.load(f)


    X_, y_ = getXYset(featureset, labels, subset_train)
    model = MLPerceptron()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x=[X_], y=[y_], batch_size=32, epochs=50, verbose=1, shuffle=True, validation_split=0.1)
    
    y_pred = model.predict([X_])
    true_ind = np.argmax(y_, axis=1)
    pred_ind = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true_ind, pred_ind)
    print("Confusion Matrix:")
    print(cm)



#    X_test, y_test = getXYset(args.featureset, args.labels, args.subset_test)