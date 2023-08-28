import os, glob 
import time
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
import concurrent.futures
from joblib import Parallel, delayed

from universal_utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '13_Totalsegmentator'
root = f'/nas124/Data_External/{dataset}'
p_names = [p for p in sorted(os.listdir(os.path.join(root,'raw'))) if os.path.isdir(os.path.join(root,'raw',p))]

img_folder = os.path.join(root,'img')
label_folder = os.path.join(root,'label')
os.makedirs(img_folder,exist_ok=True)
os.makedirs(label_folder,exist_ok=True)

with open('/home/jepark/MIAI_Segmentation/dataset/core/public_classes.json','r') as f:
    dict_class = json.load(f)
with open('/home/jepark/MIAI_Segmentation/dataset/core/to_universal/13_total_label.json','r') as f:
    original_class = json.load(f)

def get_to_rule(dict_class, original_class):
    to_rule = {}
    for original_key, original_name in original_class.items():
        for new_key, new_name in dict_class.items():
            if original_name == new_name:
                to_rule[original_name] = new_key
    return to_rule 
to_rule = get_to_rule(dict_class, original_class)



# for p_name in p_names:
def worker(root, dict_class, p_name, img_folder, label_folder):
    ct_path = os.path.join(root,'raw',p_name,'ct.nii.gz')
    ct = ''
    if not os.path.isfile(ct_path): return 
    ct = convert_nibabel_to_itk(ct_path)
    if ct is None: return 
    
    volumes = []; class_names= []
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))
    gt_arr = np.zeros_like(sitk.GetArrayFromImage(ct))
    for class_name, nvalue in to_rule.items():
        gt_path = os.path.join(root,'raw',p_name,'segmentations',f'{class_name}.nii.gz')
        gt = convert_nibabel_to_itk(gt_path)
        if gt is None: continue
        gt = sitk.GetArrayFromImage(gt)
        gt_arr[gt>0] = int(nvalue)
        volumes.append(int(np.sum((gt>0).astype(np.uint8))))
        class_names.append(class_name)
            
            
    gt_name = os.path.join(label_folder,p_name+'_label.nii.gz')
    ct_name = os.path.join(img_folder,p_name+'.nii.gz')
    if len(np.unique(gt_arr))<5:
        # print('Wrong GT 13', p_name)
        return 
    if len(np.unique(sitk.GetArrayFromImage(ct)))<1024:
        # print('Wrong CT 13', p_name)
        return 
    
    # gt = sitk.GetImageFromArray(gt_arr)
    # gt.SetSpacing(ct.GetSpacing())
    # gt.SetOrigin(ct.GetOrigin())
    # sitk.WriteImage(gt, gt_name)
    
    # ct = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct))
    # ct.SetSpacing(gt.GetSpacing())
    # ct.SetOrigin(gt.GetOrigin())
    # sitk.WriteImage(ct, ct_name)
    # print("Saved ", p_name, axcode)
    return {"image": ct_name, "label": gt_name, "volume": volumes, "class": class_name}


result = Parallel(n_jobs=4)(delayed(worker)(root, dict_class, p_name, img_folder, label_folder) for p_name in reversed(p_names))
result = list(filter(None, result))
json_saver(result, os.path.join(root,f'train_list_{dataset}.json'))