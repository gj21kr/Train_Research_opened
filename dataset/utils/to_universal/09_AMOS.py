import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '09_AMOS'
root = f'/nas124/Data_External/{dataset}'

gt_paths = glob.glob(os.path.join(root, 'raw', 'labelsTr','*'),recursive=True)
gt_paths += glob.glob(os.path.join(root, 'raw', 'labelsVa','*'),recursive=True)
gt_paths += glob.glob(os.path.join(root, 'raw', 'labelsTs','*'),recursive=True)
gt_paths = list(set([f for f in gt_paths if os.path.isfile(f)]))

os.makedirs(os.path.join(root, 'label'), exist_ok=True)

"""
    "1": "spleen", "2": "right kidney", "3": "left kidney", 
    "4": "gall bladder", "5": "esophagus", "6": "liver", 
    "7": "stomach", "8": "arota", "9": "postcava", 
    "10": "pancreas", "11": "right adrenal gland", "12": "left adrenal gland", 
    "13": "duodenum", "14": "bladder", "15": "prostate/uterus"
"""
to_rule = {
    "1": "1", "2": "2", "3": "3", 
    "4": "4", "5": "42", "6": "5", 
    "7": "6", "8": "7", 
    "10": "10", "11": "83", "12": "82", 
    "13": "56", "14": "11", "15": "12"}


with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    

# for gt_path in (gt_paths):
def worker(root, gt_path, dict_class):
    p_name = gt_path.split('/')[-1].split('.nii.gz')[0]
    ct_path = gt_path.replace('labelsTr', 'imagesTr').replace('labelsVa', 'imagesVa').replace('labelsTs', 'imagesTs')
    newGT_path = os.path.join(root, 'label', f'{p_name}.nii.gz')
    newCT_path = os.path.join(root, 'img', f'{p_name}.nii.gz')
    # Load the image
    ct = sitk.ReadImage(ct_path)
    gt = sitk.ReadImage(gt_path)
    gt_arr = sitk.GetArrayFromImage(gt)
    
    axcode = ''.join(nib.aff2axcodes(get_affine(gt)))

    label_arr = np.zeros_like(gt_arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        label_arr[gt_arr==int(ovalue)]=int(nvalue)
        volumes.append(int(np.sum((gt_arr==int(ovalue)).astype(np.uint8))))
        class_names.append(dict_class[str(nvalue)])
    
    if len(np.unique(label_arr))==0:
        print('Wrong GT 09', p_name)
        return 
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 09', p_name)
        return 
    
    newGT = sitk.GetImageFromArray(label_arr.astype(np.uint8))
    newGT.SetSpacing(gt.GetSpacing())
    newGT.SetOrigin(gt.GetOrigin())
    newGT.SetDirection(gt.GetDirection())
    sitk.WriteImage(newGT, newGT_path)
    sitk.WriteImage(ct, newCT_path)
    # print("Saved ", p_name, axcode)
    return {"image": newCT_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
    
result = Parallel(n_jobs=4)(delayed(worker)(root, gt_path, dict_class) for gt_path in gt_paths)
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))
        