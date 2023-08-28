import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataname = '10_Decathlon'
root = f'/nas124/Data_External/{dataname}'

img_folder = os.path.join(root, 'img')
label_folder = os.path.join(root, 'label')
os.makedirs(img_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

"""
    Task03_Liver
    Task05_Prostate
    Task07_Pancreas
    Task09_Spleen
"""
to_rule = {
    'Task03_Liver':5,
    'Task05_Prostate':12,
    'Task07_Pancreas':10,
    'Task09_Spleen':1,
}

with open('/home/jepark/MIAI_Segmentation/dataset/core/public_classes.json','r') as f:
    dict_class = json.load(f)
    
images = []
for dataset in list(to_rule.keys()):
    images += glob.glob(os.path.join(root, dataset,'imagesTr','*.nii.gz'))


# for dataset in list(to_rule.keys()):
def worker(root, img_folder, label_folder, img_path, dict_class):
    # images = glob.glob(os.path.join(root, dataset,'imagesTr','*.nii.gz'))
    volumes, class_names = [], []
    # for img_path in images:
    gt_path = img_path.replace('imagesTr','labelsTr')
    filename = gt_path.split('/')[-1]
    ct_path = os.path.join(img_folder, filename)
    newGT_path = os.path.join(label_folder, filename)

    # Load the image
    ct = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    gt_arr = sitk.GetArrayFromImage(gt)
    
    axcode = ''.join(nib.aff2axcodes(get_affine(gt)))

    label_arr = np.zeros_like(gt_arr)
    label_arr[gt_arr==int(1)]=int(to_rule[dataset])
    label_arr[gt_arr==int(2)]=int(103)              # tumor
    volumes.append(int(np.sum((gt_arr==int(1)).astype(np.uint8))))
    class_names.append(dict_class[str(to_rule[dataset])])

    if np.sum((gt_arr==int(2)).astype(np.uint8))>0:
        volumes.append(int(np.sum((gt_arr==int(2)).astype(np.uint8))))
        class_names.append(dict_class[str(103)])
    
    if len(np.unique(gt_arr))==0:
        print('Wrong GT 10', p_name)
        return 
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 10', p_name)
        return 
    
    
    # newGT = sitk.GetImageFromArray(label_arr.astype(np.uint8))
    # newGT.SetSpacing(gt.GetSpacing())
    # newGT.SetOrigin(gt.GetOrigin())
    # newGT.SetDirection(gt.GetDirection())
    # sitk.WriteImage(newGT, newGT_path)
    
    # sitk.WriteImage(ct, ct_path)
    # copyfile(img_path,ct_path)
    print("Saved ", filename, axcode)
    return {"image": ct_path, "label": newGT_path, "volume": volumes, "class": list(set(class_names))}
        
    
result = Parallel(n_jobs=4)(delayed(worker)(root, img_folder, label_folder, dataset, dict_class) for dataset in images) #list(to_rule.keys()))
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataname}.json'))