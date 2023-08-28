import os, sys, glob, gc, json
import cc3d
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
from batchgenerators.utilities.file_and_folder_operations import *
import warnings

from skimage import morphology

from core.utils import *
warnings.filterwarnings("ignore")

gc.collect()

p_procs = 0.1
n_procs = int(mp.cpu_count() * p_procs)

labels = [
		"Whole_Vein",
        # "PV","LGV","GCT","SMV","SV_LGEV","RGEV",
        # "PV_SV_LGV_LGEV_RGEV_GCT_SMV",
	]

with open('/home/jepark/MIAI_Data/Data_List/Vessel/20230404_PP.json', 'r') as f:
	pathes = json.load(f)
	pathes = [p for p in pathes if os.path.isdir(p)]

ct_resampling = 'trilinear'
gt_resampling = 'trilinear'
target_spacing_order = [None, None, None]
label_only = False
png_save = False
json_only = False
save_numpy = False
add_background = False
do_partial_enhancing = True
path_save = f'/raid/dataset/hutom/medical_image/vein/{ct_resampling}_{gt_resampling}/ALL_b1_cc3d6'

path_save = os.path.join(path_save, str(target_spacing_order))
path_save_ct = os.path.join(path_save, 'image')
path_save_gt = os.path.join(path_save, 'label')
if not os.path.isdir(path_save): os.makedirs(path_save, exist_ok=True)
if not os.path.isdir(path_save_ct): os.makedirs(path_save_ct, exist_ok=True)
if not os.path.isdir(path_save_gt): os.makedirs(path_save_gt, exist_ok=True)

if len(labels) == 1:
	file_name = 'binary.json' 
	gt_name = labels[0]
elif len(labels) > 1:
	file_name = 'multiClass.json' 
	gt_name = ""
	for i in labels:
		gt_name += i.split('.nii')[0]
		if i!=labels[-1]:
			gt_name += '+'
if save_numpy:
	data_type='.npy'
else:
	data_type='.nii.gz'
json_name = join(path_save, file_name)
print(gt_name, file_name, target_spacing_order, json_name)

def worker(path):
	flipx, flipy, flipz, transpose, rot = False, False, False, (1,2,0), 0
	p_name, x, ct_shape, ct_spacing = ct_loader(path)
	if p_name is None : return None
	if target_spacing_order == [None, None, None]:
		target_spacing = np.array(ct_spacing).astype(float)
	else:
		target_spacing = target_spacing_order
	print(target_spacing)
	x = orientation(x, flipx, flipy, flipz, transpose, rot)
	ct_shape = x.shape
	target_shape = list(np.rint(
		np.array(ct_shape)*np.array(ct_spacing)/np.array(target_spacing)).astype(int))
	x = resampling(x, ct_shape, ct_spacing, target_shape, target_spacing, mode=ct_resampling)
	ct_shape_new = target_shape #x.shape
	print(p_name, ct_shape, ct_shape_new)	
	
	num_classes = len(labels)
	ch_labels = np.zeros((num_classes, *ct_shape))
	volumes = []
	for i, label in enumerate(labels):
		if 'Whole' not in label and "Other" not in label and '_' in label:
			indv = label.split('_')
			volume = 0
			gt = np.zeros(ct_shape)
			for ind in reversed(indv):
				t_v, t_i = process_v3(
					path, ind, 
					flipx, flipy, flipz, transpose, rot
				)
				if t_v > 0 :
					volume += t_v
					gt += t_i
		else:
			gt = process_v3(
				path, label, 
				flipx, flipy, flipz, transpose, rot
			)
		if do_partial_enhancing==True:
			# for this_enhance in ['LGEA','RGEA','SGA','IPA','ASPDA','LGA']:
			# 	if os.path.isfile(os.path.join(path,'stor','objects',f'{this_enhance}.nii.gz')):
			# 		gt = partial_enhance(
			# 			gt, this_enhance, path, 
			# 			flipx, flipy, flipz, transpose, rot
			# 		)
			dilate_selem = morphology.ball(1)
			gt = morphology.binary_dilation(gt, dilate_selem).astype(np.uint8)
		if gt.shape == ct_shape:
			ch_labels[i] = gt
		else:
			print('Odd Shape!', p_name, ct_shape, gt.shape)
			return None
    # Resampling
	gts = []
	for gt in ch_labels:
		temp = resampling((gt>0).astype(np.uint8), ct_shape, ct_spacing, target_shape, target_spacing, mode=gt_resampling)
		gts.append(temp)
		volumes.append(int(np.sum(temp)))
	ch_labels = np.stack(gts, axis=0) # [len(labels),*ct_shape_new]
 
	# check
	if np.sum(ch_labels)==0: 
		print('Wrong GT', p_name)
		return None
		
	if len(np.unique(ch_labels))==1:
		print("Odd Data", p_name, ch_labels.shape, gt.shape)
		return None
	
	if add_background==True:
		background = np.ones((1,*ct_shape_new))
		for gt in ch_labels:
			background[0] = background[0] - gt
		ch_labels = np.concatenate([background, ch_labels], axis=0)
		ch_labels[ch_labels<0] = 0
		ch_labels[ch_labels>1] = 1
		print(p_name, ch_labels.shape, np.sum(ch_labels[0]),np.sum(ch_labels[1]))
	else:
		ch_labels[ch_labels<0] = 0
		ch_labels[ch_labels>1] = 1
		ch_labels = ch_labels[0]
		# connected_components_3d
		ch_labels = (cc3d.connected_components(ch_labels, connectivity=6)>0).astype(np.uint8)

	# Save
	ct_name, label_name = last_work(
		path_save_ct, path_save_gt, p_name, gt_name, x, ch_labels, 
		target_spacing,  
		json_only, label_only, data_type, save_numpy, png_save
	)
	return {"image": ct_name, "label": label_name, "volume": volumes, "HU [min,max]": [int(np.min(x)), int(np.max(x))]}


if __name__ == '__main__':
	procs = []

	# div = len(pathes) // n_procs
	# jobs_per_proc = [pathes[i*div : (i+1)*div] if i!=n_procs-1 else pathes[i*div:] for i in range(n_procs)]
	# pool = mp.Pool(processes=n_procs)
	# results = pool.map(worker, pathes)
	results = Parallel(n_jobs=n_procs)(delayed(worker)(i) for i in pathes)
	results = list(filter(None, results))

	# results = []
	# for path in pathes:
	# 	results.append(worker(path))
	
	json_saver(results, json_name)