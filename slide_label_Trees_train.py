import json, os, sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, average_precision_score, precision_score, make_scorer, recall_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint


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
        if labels[s]==0 and np.random.rand()>=2:
            continue
        fs = featureset[s]
        x = []
        for feature in FEATURE_ORDER:
            x.append(fs[feature])
        X_train.append(x)
        l = [0,0,0,0]
        l[labels[s]] = 1
        y_train.append(labels[s])
    
    return shuffle(X_train, y_train)

def printCM(clf, y_true, X_test, txt='Train'):
    y_pred = clf.predict(X_test)
    true_ind = np.argmax(y_true, axis=1)
    pred_ind = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true_ind, pred_ind)
    print(txt, "Confusion Matrix:")
    print(cm)



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
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3)

    #clf = tree.DecisionTreeClassifier()
    #clf = RandomForestClassifier()
    '''
    clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50), n_estimators=100)
    clf = clf.fit(X_train, y_train)

    print("\nAdaBoost Precision Scores:")
    print("Micro: ", precision_score(y_test, clf.predict(X_test), average='micro'))
    print("None: ", precision_score(y_test, clf.predict(X_test), average=None))


    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)

    print("\nRandomForest Precision Scores:")
    print("Micro: ", precision_score(y_test, clf.predict(X_test), average='micro'))
    print("None: ", precision_score(y_test, clf.predict(X_test), average=None))
    '''

#    print("\nCV Ada:\n", cross_val_score(AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50), n_estimators=100), X_, y_, scoring='precision_micro', cv=3))
#    print("\nCV RandomForest:\n", cross_val_score(RandomForestClassifier(n_estimators=50), X_, y_, scoring='precision_micro', cv=3))

    scoring = {'prec_global': 'precision_micro',
    'prec_negative': make_scorer(precision_score, average=None, labels=[0]),
    'prec_itc': make_scorer(precision_score, average=None, labels=[1]),
    'prec_micro': make_scorer(precision_score, average=None, labels=[2]),
    'prec_macro': make_scorer(precision_score, average=None, labels=[3]),
    'rec_global': 'recall_micro',
    'rec_negative': make_scorer(recall_score, average=None, labels=[0]),
    'rec_itc': make_scorer(recall_score, average=None, labels=[1]),
    'rec_micro': make_scorer(recall_score, average=None, labels=[2]),
    'rec_macro': make_scorer(recall_score, average=None, labels=[3])
    }

    
    '''
    score_ada = cross_validate(AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50, class_weight='balanced'), n_estimators=100), X_, y_, scoring=scoring, cv=3, return_estimator=True)
    score_rf = cross_validate(RandomForestClassifier(n_estimators=50, class_weight='balanced'), X_, y_, scoring=scoring, cv=3, return_estimator=True)

    print("\nCV AdaBoost:\n")
    pprint(score_ada)

    print("\nCV RandomForest:\n")
    pprint(score_rf)

    print('\nADA Feature importance:') 
    for k,v in zip(FEATURE_ORDER, score_ada['estimator'][0].feature_importances_):
        print(k, "\t:\t", v)
    print('\nRandomForest Feature importance:') 
    for k,v in zip(FEATURE_ORDER, score_rf['estimator'][0].feature_importances_):
        print(k, "\t:\t", v)
    '''

    clf = RandomForestClassifier(n_estimators=50, class_weight='balanced')
    param_grid = [
        {'max_depth':[3,5,10,None], 'min_samples_split':[2,5,9,14], 'min_samples_leaf':[1,3,5,7,11]}
    ]
    gscv = GridSearchCV(clf, param_grid, scoring=scoring, cv=3, return_train_score=True, refit=False)
    gscv.fit(X_, y_)

    pprint(gscv.cv_results_)



    #printCM(clf, y_train, X_train)

    #printCM(clf, y_test, X_test, 'Test')



#    X_test, y_test = getXYset(args.featureset, args.labels, args.subset_test)