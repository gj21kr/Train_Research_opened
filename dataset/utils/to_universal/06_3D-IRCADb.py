import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '06_Liver segmentation 3D-IRCADb'
root = f'/nas124/Data_External/{dataset}'

dicom_paths = glob.glob(os.path.join(root, '**', 'PATIENT_DICOM','*'),recursive=True)
dicom_paths = [f for f in dicom_paths if os.path.isfile(f)]
dicom_paths = sorted(list(set([os.path.dirname(path) for path in dicom_paths])))

os.makedirs(os.path.join(root, 'img'), exist_ok=True)
os.makedirs(os.path.join(root, 'label'), exist_ok=True)
os.makedirs(os.path.join(root, 'orig_label'), exist_ok=True)

to_rule = {
    "spleen": 1, "rightkidney":2, "leftkidney":3, 
    "rightlung": 86, "leftlung": 87, "liver": 5, 
    "tumor":103, "cyst":104
}

with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    
def load_label(label_path):    
    label_reader = sitk.ImageSeriesReader()
    label_names = label_reader.GetGDCMSeriesFileNames(label_path)
    label_reader.SetFileNames(label_names)
    label_image = label_reader.Execute()
    return (sitk.GetArrayFromImage(label_image)>0)

# for dicom_path in dicom_paths:
def worker(root, dicom_path, dict_class):
    parent_folder = os.path.abspath(os.path.join(dicom_path, os.pardir))
    p_name = parent_folder.split('/')[-1].split('.')[-1]

    newCT_path = os.path.join(root, 'img', f'image_{p_name}.nii.gz')
    gt_path = os.path.join(root, 'orig_label', f'mask_{p_name}.nii.gz')
    newGT_path = os.path.join(root, 'label', f'mask_{p_name}.nii.gz')

    # Load the DICOM series
    dicom_reader = sitk.ImageSeriesReader()
    dicom_names = dicom_reader.GetGDCMSeriesFileNames(dicom_path)
    dicom_reader.SetFileNames(dicom_names)
    ct = dicom_reader.Execute()
    axcode = ''.join(nib.aff2axcodes(get_affine(ct)))

    gt_folders = glob.glob(os.path.join(parent_folder, 'MASKS_DICOM','*'),recursive=True)
    label_arr = np.zeros_like(sitk.GetArrayFromImage(ct))
    volumes, class_names = [], []
    for gt_folder in gt_folders:
        filename = gt_folder.split('/')[-1]
        arr = load_label(gt_folder)
        if not label_arr.shape == arr.shape:
            print('Wrong Size!', p_name, filename)
            continue
        gt = sitk.GetImageFromArray(arr.astype(np.uint8))
        gt.SetSpacing(ct.GetSpacing())
        gt.SetOrigin(ct.GetOrigin())
        gt.SetDirection(ct.GetDirection())
        sitk.WriteImage(gt, gt_path.replace('.nii.gz',f'_{filename}.nii.gz'))
        
        nvalue = 0
        if filename in list(to_rule.keys()):
            nvalue = int(to_rule[filename])
            label_arr[arr>0] = nvalue
        elif 'livertumor' in filename:
            nvalue = int(to_rule["tumor"])
            label_arr[arr>0] = nvalue
        elif 'liverkyst' in filename:
            nvalue = int(to_rule["cyst"])
            label_arr[arr>0] = nvalue  
        if np.sum((arr>0).astype(np.uint8))>0 and nvalue != 0:
            volumes.append(int(np.sum((arr>0).astype(np.uint8))))
            class_names.append(dict_class[str(nvalue)])
     
    if len(np.unique(label_arr))==0:
        print('Wrong GT 06', p_name)
        return
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 06', p_name)
        return 
    
    gt = sitk.GetImageFromArray(label_arr.astype(np.uint8))
    gt.SetSpacing(ct.GetSpacing())
    gt.SetOrigin(ct.GetOrigin())
    gt.SetDirection(ct.GetDirection())
    sitk.WriteImage(gt, newGT_path)
    sitk.WriteImage(ct, newCT_path)
    # print("Saved ", p_name, axcode)
    return {"image": newCT_path, "label": newGT_path, "volume": volumes, "class": class_names}
        
    
result = Parallel(n_jobs=4)(delayed(worker)(root, dicom_path, dict_class) for dicom_path in dicom_paths)
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))
        