import os, glob 
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from universal_utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '08_AbdomenCT-1K'
root = f'/nas124/Data_External/{dataset}'

gt_paths = glob.glob(os.path.join(root, 'raw', 'Mask', '*.nii.gz'),recursive=True)
gt_paths = list(set([f for f in gt_paths if os.path.isfile(f)]))

os.makedirs(os.path.join(root, 'img'), exist_ok=True)
os.makedirs(os.path.join(root, 'label'), exist_ok=True)

"""
    0: background
    1: liver 
    2: kidney
    3: spleen
    4: pancreas
"""
to_rule = {
    1:5, 2:'2,3', 3:1, 4:10
}


with open('/home/jepark/MIAI_Segmentation/dataset/utils/public_classes.json','r') as f:
    dict_class = json.load(f)
    

# for gt_path in (gt_paths):
def worker(root, gt_path, dict_class):
    p_name = gt_path.split('/')[-1].split('.nii.gz')[0]
    ct_path = gt_path.replace('Mask', 'image').replace('.nii.gz', '_0000.nii.gz')
    
    newGT_path = os.path.join(root, 'label', f'{p_name}.nii.gz')
    newCT_path = os.path.join(root, 'img', f'{p_name}_0000.nii.gz')

    # Load the image
    ct = sitk.ReadImage(ct_path)
    gt = sitk.ReadImage(gt_path)
    # print('08', p_name, np.unique(sitk.GetArrayFromImage(ct)))
    gt_arr = sitk.GetArrayFromImage(gt)
    
    axcode = ''.join(nib.aff2axcodes(get_affine(gt)))
    if axcode == 'RPI':
        gt_arr = gt_arr[:, ::-1, :]
        gt.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    else:
        if not axcode=='LPS':
            print('check this direction', axcode, p_name)
            return
        
    label_arr = np.zeros_like(gt_arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        if type(nvalue) is str:
            # split left and right kidney
            whole = np.zeros_like(gt_arr)
            whole[gt_arr==ovalue] = 1
            shape = whole.shape 
            whole[:,:,:shape[2]//2] *= 2
            whole[:,:,shape[2]//2:] *= 3
            label_arr[whole==2]=3
            label_arr[whole==3]=2
            if np.sum((whole==2).astype(np.uint8))>0:
                volumes.append(int(np.sum((whole==2).astype(np.uint8))))
                class_names.append(dict_class[str(2)])
                
            if np.sum((whole==3).astype(np.uint8))>0:
                volumes.append(int(np.sum((whole==3).astype(np.uint8))))
                class_names.append(dict_class[str(3)])
        else:
            label_arr[gt_arr==ovalue]=nvalue          
            if np.sum((gt_arr==ovalue).astype(np.uint8))>0:
                volumes.append(int(np.sum((gt_arr==ovalue).astype(np.uint8))))
                class_names.append(dict_class[str(nvalue)])
    
    if len(np.unique(label_arr))==0:
        print('Wrong GT 08', p_name)
        return
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 08', p_name)
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
        