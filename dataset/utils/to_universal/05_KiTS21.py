import os, glob 
import shutil 

original = '/nas124/Data_External/05_KiTS21'
root = '/nas124/Data_External/05_KiTS21'
os.makedirs(os.path.join(root,'img'),exist_ok=True)
os.makedirs(os.path.join(root,'label'),exist_ok=True)

for case in glob.glob(os.path.join(original, '*')):
    if not os.path.isdir(case): continue
    name = case.split('/')[-1]; print(name)
    name = name.split('_')[-1][1:]
    ct_image = os.path.join(case, 'imaging.nii.gz')
    ct_new = os.path.join(root, 'img', f'img{name}.nii.gz')
    gt_image = os.path.join(case, 'aggregated_MAJ_seg.nii.gz')
    gt_new = os.path.join(root, 'label', f'label{name}.nii.gz')
    shutil.copyfile(ct_image, ct_new)
    shutil.copyfile(gt_image, gt_new)