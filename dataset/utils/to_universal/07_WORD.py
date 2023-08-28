import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '07_WORD'
root = f'/nas124/Data_External/{dataset}'

dicom_paths = glob.glob(os.path.join(root, 'raw', 'imagesTr','*'),recursive=True)
dicom_paths += glob.glob(os.path.join(root, 'raw', 'imagesVal','*'),recursive=True)
dicom_paths += glob.glob(os.path.join(root, 'raw', 'imagesTs','*'),recursive=True)
dicom_paths = list(set([f for f in dicom_paths if os.path.isfile(f)]))

os.makedirs(os.path.join(root, 'img'), exist_ok=True)
os.makedirs(os.path.join(root, 'label'), exist_ok=True)

"""
    0: background
    1: liver 
    2: spleen
    3: kidney(L)
    4: kidney(R)
    5: stomach
    6: gallbladder
    7: esophagus
    8: pancreas
    9: cuodenum
    10:colon
    11:intestine
    12:adrenal
    13:rectum
    14:bladder
    15:head of femur(L)
    16:head of femur(R)
"""
to_rule = {
    1:5, 2:1, 3:3, 4:2, 5:6, 6:4, 7:42, 8:10, 9:56, 10:57, 
    11:55, 14:11
}

with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    

# for dicom_path in dicom_paths:
def worker(root, dicom_path, dict_class):
    p_name = dicom_path.split('/')[-1].split('.nii.gz')[0]

    ct_path = os.path.join(root, 'img', f'{p_name}.nii.gz')
    gt_path = os.path.join(root, 'orig_label', f'{p_name}.nii.gz')
    newGT_path = os.path.join(root, 'label', f'{p_name}.nii.gz')

    # Load the DICOM series
    ct = sitk.ReadImage(dicom_path)
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))
    
    if 'imagesTr' in dicom_path:
        gt_path = dicom_path.replace('imagesTr','labelsTr')
    elif 'imagesTs' in dicom_path:
        gt_path = dicom_path.replace('imagesTs','labelsTs')
    elif 'imagesVal' in dicom_path:
        gt_path = dicom_path.replace('imagesVal','labelsVal')
    
    label_arr = np.zeros_like(sitk.GetArrayFromImage(ct))
    gt = sitk.ReadImage(gt_path)
    gt_arr = sitk.GetArrayFromImage(gt)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        label_arr[gt_arr==ovalue]=nvalue    
        if np.sum((gt_arr==ovalue).astype(np.uint8))>0:
            volumes.append(int(np.sum((gt_arr==ovalue).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
    
    if len(np.unique(label_arr))==0:
        print('Wrong GT 07', p_name)
        return
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 07', p_name)
        return 
    
    
    gt = sitk.GetImageFromArray(label_arr.astype(np.uint8))
    gt.SetSpacing(ct.GetSpacing())
    gt.SetOrigin(ct.GetOrigin())
    gt.SetDirection(ct.GetDirection())
    sitk.WriteImage(gt, newGT_path)
    sitk.WriteImage(ct, ct_path)
    # print("Saved ", p_name, axcode)
    return {"image": ct_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
    
result = Parallel(n_jobs=4)(delayed(worker)(root, dicom_path, dict_class) for dicom_path in dicom_paths)
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))
        