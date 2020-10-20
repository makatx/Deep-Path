from Slide import Slide
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import json

folder = '/home/mak/PathAI/slides/train/'

'''
slides_list = 'slide_lists/200808_annotated_slides_TRAIN_SET.json'
filenames = []
with open(slides_list, 'r') as f:
    filenames = json.load(f)
extraction_level = 6
view_level = 1
skip_negatives = True
thresh_method = 'GRAY'
save_output_to = 'patch_lists/201019_PCL_TmpToProcess_AnnotTrainSet_FractionLabel_L1_EL6_skipNegatives.json'
'''

'''
slides_list = 'slide_lists/200808_annotated_slides_TRAIN_SET.json'
filenames = []
with open(slides_list, 'r') as f:
    filenames = json.load(f)
extraction_level = 10
view_level = 1
skip_negatives = False
thresh_method = 'GRAY'
save_output_to = 'patch_lists/201019_PCL_TmpToProcess_AnnotTrainSet_FractionLabel_L1_ELmax_WNegatives.json'


'''

slides_list = 'slide_lists/200808_slide_label_train_dict.json'
filenames = []
with open(slides_list, 'r') as f:
    filenames = json.load(f)['negative']
extraction_level = 10
view_level = 1
skip_negatives = False
thresh_method = 'GRAY'
save_output_to = 'patch_lists/201019_PCL_TmpToProcess_NegativeTrainSet_FractionLabel_L1_ELmax.json'


'''
filenames = ['patient_021_node_3.tif',
'patient_004_node_4.tif',
'patient_022_node_3.tif',
'patient_024_node_2.tif',
'patient_037_node_1.tif',
'patient_045_node_1.tif',
'patient_050_node_3.tif',
'patient_061_node_4.tif',
'patient_064_node_4.tif',
'patient_089_node_3.tif']
'''

pcl_negative = []
pcl_annot = []
pcl_neigh = []

for filename in tqdm(filenames):
    prefix, _ = os.path.splitext(filename)
    annotn_filename = prefix + '.xml'
    annot = False

    if os.path.exists(folder+annotn_filename):
        #print("processing {} with {}". format(filename, annotn_filename))
        slide = Slide(folder+filename, folder+annotn_filename, extraction_level=extraction_level)
        pcl = slide.getPatchCoordListWLabels(thresh_method=thresh_method, with_filename=True, view_level=view_level, skip_negatives=skip_negatives)
        
        if not skip_negatives:
            pcl_negative.extend(pcl[0])
        pcl_annot.extend(pcl[1])
        pcl_neigh.extend(pcl[2])

    elif not skip_negatives:
        #print("processing {}". format(filename))
        slide = Slide(folder+filename, extraction_level=extraction_level)
        #annot = True
        pcl = slide.getPatchCoordListWLabels(thresh_method=thresh_method, with_filename=True, view_level=view_level)
        
        pcl_negative.extend(pcl[0])

pcl_dict = {}

if not skip_negatives:
    pcl_dict['negative'] = pcl_negative
pcl_dict['neighbor'] = pcl_neigh
pcl_dict['annotation'] = pcl_annot

with open(save_output_to, 'w') as f:
    json.dump(pcl_dict, f)