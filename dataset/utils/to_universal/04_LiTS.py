import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '04_LiTS'
root = f'/nas124/Data_External/{dataset}'

files = glob.glob(os.path.join(root, 'raw', 'segmentations', '*.nii'), recursive=True)

to_rule = {"1":"5","2":"103"}

with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    
# for file in files:
def worker(root, file, dict_class):
    filename = file.split('/')[-1].replace('nii','nii.gz')
    
    ct = sitk.ReadImage(file.replace('segmentation','volume'))
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))
    
    img = sitk.ReadImage(file)
    arr = sitk.GetArrayFromImage(img)
    
    newCT_path = os.path.join(root,'img',filename)
    newGT_path = os.path.join(root,'label',filename)
    
    temp = np.zeros_like(arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        temp[arr==int(ovalue)] = int(nvalue) 
        if np.sum((arr==int(ovalue)).astype(np.uint8))>0:
            volumes.append(int(np.sum((arr==int(ovalue)).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
        
    if len(np.unique(temp))==0:
        print('Wrong GT 08', filename)
        return
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 08', filename)
        return 
    
    arr = sitk.GetImageFromArray(temp.astype(np.uint8))    
    arr.SetSpacing(ct.GetSpacing())
    arr.SetOrigin(ct.GetOrigin())
    arr.SetDirection(ct.GetDirection())
    sitk.WriteImage(arr, newGT_path)
    sitk.WriteImage(ct, newCT_path)
    # print("Saved ", filename, axcode)
    return {"image": newCT_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
result = Parallel(n_jobs=4)(delayed(worker)(root, file, dict_class) for file in files)
result = list(filter(None, result))
json_saver(result, os.path.join(root,f'train_list_{dataset}.json'))