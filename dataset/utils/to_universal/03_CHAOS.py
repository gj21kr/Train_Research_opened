import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from universal_utils import get_affine, convert_nibabel_to_itk, json_saver

from PIL import Image

dataset = '03_CHAOS'
root = f'/nas124/Data_External/{dataset}'

folders = sorted(glob.glob(os.path.join(root, 'Train_Sets', 'CT', '*')))

os.makedirs(os.path.join(os.path.join(root, 'img')),exist_ok=True)
os.makedirs(os.path.join(os.path.join(root, 'label')),exist_ok=True)
os.makedirs(os.path.join(os.path.join(root, 'orig_label')),exist_ok=True)


to_rule = {"1": "5"}


with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    
    
# for folder in folders:
def worker(root, folder, dict_class):
    p_name = folder.split('/')[-1]
    dicom_path = os.path.join(folder,'DICOM_anon')
    png_path = os.path.join(folder,'Ground')
    
    # Load the DICOM series
    dicom_reader = sitk.ImageSeriesReader()
    dicom_names = dicom_reader.GetGDCMSeriesFileNames(dicom_path)
    dicom_reader.SetFileNames(dicom_names)
    dicom_image = dicom_reader.Execute()
    axcode = ''.join(nib.aff2axcodes(get_affine(dicom_image)))
    
    # # Load the GT images
    files = sorted(glob.glob(os.path.join(png_path,'*.png')))[::-1]
    gt_image = []
    for file in files:
        gt_image.append(np.array(Image.open(file), dtype=np.uint8))
    gt_image = np.stack(gt_image)
    
    # convert value 
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        newGT = np.zeros_like(gt_image)
        newGT[gt_image>0] = int(nvalue)
        if np.sum((gt_image>0).astype(np.uint8))>0:
            volumes.append(int(np.sum((gt_image>0).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
    
    if len(np.unique(newGT))==0:
        print('Wrong GT 03', p_name)
        return 
    
    # gt_image = sitk.GetImageFromArray(gt_image.astype(np.uint8))
    # gt_image.SetSpacing(dicom_image.GetSpacing())
    # gt_image.SetOrigin(dicom_image.GetOrigin())
    # gt_image.SetDirection(dicom_image.GetDirection())
    newGT = sitk.GetImageFromArray(newGT.astype(np.uint8))
    newGT.SetSpacing(dicom_image.GetSpacing())
    newGT.SetOrigin(dicom_image.GetOrigin())
    newGT.SetDirection(dicom_image.GetDirection())

    # Save the modified NIfTI file 
    ct_path = os.path.join(root, 'img', f'{p_name}_image.nii.gz')
    gt_path2 = os.path.join(root, 'label', f'{p_name}_segmentation.nii.gz')
    # gt_path1 = os.path.join(root, 'orig_label', f'{p_name}_segmentation.nii.gz')

    sitk.WriteImage(dicom_image, ct_path)
    # sitk.WriteImage(gt_image, gt_path1)    
    sitk.WriteImage(newGT, gt_path2)    
    # print("Saved ", p_name, axcode)
    return {"image": ct_path, "label": gt_path2, "volume": volumes, "class": class_names}
        
result = Parallel(n_jobs=4)(delayed(worker)(root, folder, dict_class) for folder in folders)
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))
        