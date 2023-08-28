import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from universal_utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '02_TCIA_Pancreas-CT'
root = f'/nas124/Data_External/{dataset}'

os.makedirs(os.path.join(root, 'img'), exist_ok=True)
os.makedirs(os.path.join(root, 'label'), exist_ok=True)

dicom_paths = glob.glob(os.path.join(root, '**', '*.dcm'),recursive=True)
dicom_paths = list(set([os.path.dirname(path) for path in dicom_paths]))

to_rule = {"1": "10"}

with open('/home/jepark/MIAI_Segmentation/dataset/core/public_classes.json','r') as f:
    dict_class = json.load(f)
    

# for dicom_path in dicom_paths:
def worker(root, dicom_path, dict_class):
    p_name = dicom_path.split(root)[-1].split('/')[3]
    
    newCT_path = os.path.join(root, 'img', f'{p_name}.nii.gz')
    newGT_path = os.path.join(root, 'label', f'{p_name}.nii.gz')
    # Load the DICOM series
    dicom_reader = sitk.ImageSeriesReader()
    dicom_names = dicom_reader.GetGDCMSeriesFileNames(dicom_path)
    dicom_reader.SetFileNames(dicom_names)
    ct = dicom_reader.Execute()    
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))

    p_num = p_name.split('_')[-1]
    gt_path = os.path.join(root, 'raw', 'label', f'label{p_num}.nii.gz')
    # Load the Data 
    gt = sitk.ReadImage(gt_path)
    gt_arr = sitk.GetArrayFromImage(gt)
    
    temp = np.zeros_like(gt_arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        temp[gt_arr==int(ovalue)] = int(nvalue)
        if np.sum((gt_arr==int(ovalue)).astype(np.uint8))>0:
            volumes.append(int(np.sum((gt_arr==int(ovalue)).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
     
    if len(np.unique(temp))==0:
        print('Wrong GT 02', p_name)
        return
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 02', p_name)
        return 
    
    
    # Save the modified NIfTI file
    sitk.WriteImage(ct, newCT_path)
    newGT=sitk.GetImageFromArray(temp.astype(np.uint8))
    newGT.SetSpacing(gt.GetSpacing())
    newGT.SetOrigin(gt.GetOrigin())
    newGT.SetDirection(gt.GetDirection())
    sitk.WriteImage(newGT, newGT_path)
    # print("Saved ", p_name, axcode)
    return {"image": newCT_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
    
result = Parallel(n_jobs=4)(delayed(worker)(root, dicom_path, dict_class) for dicom_path in dicom_paths)
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))
        