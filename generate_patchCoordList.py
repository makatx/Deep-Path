from Slide import Slide
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import json

folder = '/home/mak/PathAI/slides/train/'
filenames = []
for f in os.listdir(folder):
    if f.endswith('.tif'):
        filenames.append(f)

pcl_negative = []
pcl_annot = []
pcl_neigh = []

for filename in tqdm(filenames):
    prefix, _ = os.path.splitext(filename)
    annotn_filename = prefix + '.xml'
    annot = False

    if os.path.exists(folder+annotn_filename):
        #print("processing {} with {}". format(filename, annotn_filename))
        slide = Slide(folder+filename, folder+annotn_filename)
        pcl = slide.getPatchCoordListWLabels(thresh_method='OTSU', with_filename=True, view_level=0)
        
        pcl_negative.extend(pcl[0])
        pcl_annot.extend(pcl[1])
        pcl_neigh.extend(pcl[2])

    else:
        #print("processing {}". format(filename))
        slide = Slide(folder+filename)
        annot = True
        pcl = slide.getPatchCoordListWLabels(thresh_method='OTSU', with_filename=True, view_level=0)
        
        pcl_negative.extend(pcl[0])

pcl_dict = {}
pcl_dict['negative'] = pcl_negative
pcl_dict['neighbor'] = pcl_neigh
pcl_dict['annotation'] = pcl_annot

with open('200421_patch_coord_lists_w_labels_level0.json', 'w') as f:
    json.dump(pcl_dict, f)