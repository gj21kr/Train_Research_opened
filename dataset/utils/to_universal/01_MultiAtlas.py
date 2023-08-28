import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from universal_utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '01_Multi-Atlas_Labeling'
root = f'/nas124/Data_External/{dataset}'

files = glob.glob(os.path.join(root, 'Abdomen','RawData','Training','label','*.nii.gz'))

to_rule = {
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "42",
        "6": "5",
        "7": "6",
        "8": "7",
        "9": "8",
        "10": "9",
        "11": "10",
        "12": "83",
        "13": "84"
    }


with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    
    
# for f in files:
def worker(root, f, dict_class):
    filename = f.split('/')[-1].split('.nii')[0]
    newCT_path = os.path.join(root,"img",f"{filename}.nii.gz")
    newGT_path = os.path.join(root,"label",f"{filename}.nii.gz")
    # Load the Data 
    ct = sitk.ReadImage(f.replace('label','img'))
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))
    gt = sitk.ReadImage(f)
    arr = sitk.GetArrayFromImage(gt)
    
    temp = np.zeros_like(arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        temp[arr==int(ovalue)] = int(nvalue) 
        if np.sum((arr==int(ovalue)).astype(np.uint8))>0:
            volumes.append(int(np.sum((arr==int(ovalue)).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
         
    if len(np.unique(arr))==0:
        print('Wrong GT 01', p_name)
        return 
    
    newGT=sitk.GetImageFromArray(temp.astype(np.uint8))
    newGT.SetSpacing(gt.GetSpacing())
    newGT.SetOrigin(gt.GetOrigin())
    newGT.SetDirection(gt.GetDirection())
    sitk.WriteImage(newGT,newGT_path)
    sitk.WriteImage(ct,newCT_path)
    # print("Saved ", filename, axcode)
    return {"image": newCT_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
result = Parallel(n_jobs=4)(delayed(worker)(root, f, dict_class) for f in files)
result = list(filter(None, result))
json_saver(result, os.path.join(root,f'train_list_{dataset}.json'))