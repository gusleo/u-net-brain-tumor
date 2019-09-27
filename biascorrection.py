import tensorlayer as tl
import numpy as np
import os
import csv
import random
import gc
import pickle
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm


HGG_bias_data_path = "data/Brats17TrainingData_Bias/HGG"
LGG_bias_data_path = "data/Brats17TrainingData_Bias/LGG"

if not os.path.exists(HGG_bias_data_path):
    os.makedirs(HGG_bias_data_path)

if not os.path.exists(LGG_bias_data_path):
    os.makedirs(LGG_bias_data_path)


HGG_data_path = "data/Brats17TrainingData/HGG"
LGG_data_path = "data/Brats17TrainingData/LGG"

HGG_path_list = tl.files.load_folder_list(path=HGG_data_path)
LGG_path_list = tl.files.load_folder_list(path=LGG_data_path)

HGG_len = len(HGG_path_list)
LGG_len = len(LGG_path_list)

print("Data training used for HGG: {} and LGG: {}".format(
    HGG_len, LGG_len))  # 210 #75
# print(len(HGG_path_list), len(LGG_path_list)) #210 #75

HGG_name_list = [os.path.basename(p) for p in HGG_path_list]
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]

data_types = ['flair', 't1', 't1ce', 't2']
print("LOAD ALL IMAGES' PATH AND COMPUTE MEAN/ STD\n============================")
for j in tqdm(HGG_name_list):
    for i in data_types:
        img_path = os.path.join(HGG_data_path, j, j + '_' + i + '.nii.gz')
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        data = sitk.GetArrayFromImage(img)
        img_data = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img_data, 0, 0))
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
        new_img = os.path.join(HGG_bias_data_path, j, j + '_' + i + '.nii.gz')
        sitk.WriteImage(corrected_img, new_img)

for j in tqdm(LGG_name_list):
    for i in data_types:
        img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        data = sitk.GetArrayFromImage(img)
        img_data = sitk.Cast(img, sitk.sitkFloat32)
        img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img_data, 0, 0))
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
        new_img = os.path.join(LGG_bias_data_path, j, j + '_' + i + '.nii.gz')
        sitk.WriteImage(corrected_img, new_img)
