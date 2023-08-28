import os, sys, glob, gc
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *	
from skimage.restoration import denoise_nl_means

try:
    from dcm_reader import Hutom_DicomLoader
except:
    import ultraimport
    Hutom_DicomLoader = ultraimport('../dcm_reader.py', 'Hutom_DicomLoader')


def get_affine(gt):
    direction_matrix = gt.GetDirection()
    spacing = gt.GetSpacing()
    origin = gt.GetOrigin()
    # Create the affine matrix
    affine_matrix = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            affine_matrix[i, j] = direction_matrix[i * 3 + j] * spacing[j]
        affine_matrix[i, 3] = origin[i]
    affine_matrix[3, 3] = 1.0
    return affine_matrix 


def convert_nibabel_to_itk(path, return_numpy=False):
    try:
        # Load the NIfTI image using nibabel
        nifti_image = nib.load(path)

        # Get the image data and affine transformation matrix
        data = nifti_image.get_fdata().transpose(2,1,0)[:,::-1,::-1]
        affine = nifti_image.affine
        if return_numpy==True:
            return data

        # Convert the data to SimpleITK image
        itk_image = sitk.GetImageFromArray(data)

        # Set the image origin and spacing using the affine transformation matrix
        itk_image.SetOrigin(affine[:3, 3])
        itk_image.SetSpacing(np.abs(affine.diagonal()[:3]))
		# Compute the direction matrix from the affine
        rotation_matrix = affine[:3, :3]
        direction_matrix = rotation_matrix / np.linalg.norm(rotation_matrix, axis=0)
        itk_image.SetDirection(direction_matrix.ravel())
        return itk_image
    except:
        return None
    
def json_saver(results, json_name):
	results = [r for r in results if not r is None]
	results = sorted(results, key=lambda item:item['image'])
	json_dict = OrderedDict()
	json_dict['numTraining'] = len(results)
	json_dict['numTest'] = len([])
	json_dict['training'] = results
	json_dict['test'] = []

	save_json(json_dict, json_name); os.chmod(json_name, 0o777)
	print('done', json_name)
