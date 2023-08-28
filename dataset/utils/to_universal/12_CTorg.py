import os, glob
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from shutil import copyfile
from joblib import Parallel, delayed

from utils import get_affine, convert_nibabel_to_itk, json_saver

dataset = '12_CT-ORG'
root = f'/nas124/Data_External/{dataset}'

img_folder = os.path.join(root, 'img')
label_folder = os.path.join(root, 'label')
os.makedirs(img_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

images = glob.glob(os.path.join(root, 'orig_img','*.nii.gz'))
"""
0: Background (None of the following organs)
1: Liver
2: Bladder
3: Lungs
4: Kidneys
5: Bone
6: Brain
"""
to_rule = {
    1: 5,
    2: 11,
    3: '46,47',
    4: '2,3'
}


with open('./public_classes.json','r') as f:
    dict_class = json.load(f)
    
# for img_path in images:
def worker(img_path, img_folder, label_folder, dict_class):
    gt_path = img_path.replace('orig_img','orig_label').replace('volume-','labels-')
    filename = gt_path.split('/')[-1]
    ct_path = os.path.join(img_folder, filename.replace('labels-','volume-'))
    newGT_path = os.path.join(label_folder, filename)

    # Load the image
    ct = sitk.ReadImage(img_path)      
    gt = sitk.ReadImage(gt_path)
    gt_arr = sitk.GetArrayFromImage(gt)
    axcode = ''.join(nib.aff2axcodes(get_affine(gt)))
    if axcode[1]=='P':
        gt_arr = gt_arr[:,:,::-1]
        spacing = ct.GetSpacing()
        ct = sitk.GetImageFromArray(sitk.GetArrayFromImage(ct)[:,:,::-1])
        ct.SetSpacing(spacing)
        ct.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

    label_arr = np.zeros_like(gt_arr)
    volumes, class_names = [], []
    for ovalue, nvalue in to_rule.items():
        if type(nvalue) is str:
            # split left and right kidney
            whole = np.zeros_like(gt_arr)
            whole[gt_arr==ovalue] = 1
            shape = whole.shape 
            whole[:,:,:shape[2]//2] *= int(nvalue.split(',')[1])
            whole[:,:,shape[2]//2:] *= int(nvalue.split(',')[0])
            label_arr[whole==2]=2
            label_arr[whole==3]=3
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
        print('Wrong GT 12', p_name)
        return 
    if len(np.unique(sitk.GetArrayFromImage(ct)))==0:
        print('Wrong CT 12', p_name)
        return 
    
    
    newGT = sitk.GetImageFromArray(label_arr.astype(np.uint8))
    newGT.SetSpacing(ct.GetSpacing())
    newGT.SetOrigin(ct.GetOrigin())
    newGT.SetDirection(ct.GetDirection())
    sitk.WriteImage(newGT, newGT_path)
    sitk.WriteImage(ct, ct_path)
    # print("Saved ", filename, axcode)
    return {"image": ct_path, "label": newGT_path, "volume": volumes, "class": class_names}

    
result = Parallel(n_jobs=4)(delayed(worker)(img_path, img_folder, label_folder, dict_class) for img_path in reversed(images))
result = list(filter(None, result))
json_saver(result, os.path.join(root, f'train_list_{dataset}.json'))