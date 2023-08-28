import math
import numpy 
from monai.transforms import *
import custom_transforms as ct
import custom_intensity as ci

def call_transforms(config):
	## train transforms
	train_transforms = [
		LoadImaged(keys=["image", "label"], reader='ITKReader', image_only=True),
		ct.CheckOrientationd(keys=["image","label"], transpose=(1,0,2)),
		# ct.CheckSize(bool_raise=False), 
		ct.PresetToClass(
				 preset=config["universal_class_preset"],
    	),
		ct.ClassModify(
                 pres=config["class_modify"]["pres"],
                 posts=config["class_modify"]["posts"]
                 ),
		ci.ZNormalizationd(keys=["image"], contrast=config["CONTRAST"], clip=True),
		ct.CheckDimd(keys=["image","label"], type_="channel_first"), 
		Spacingd(
    			keys=["image","label"], pixdim=config["SPACING"], 
       			mode=config["Interpolation"], align_corners=True),
	]    
	val_transforms = train_transforms.copy()

	train_transforms += [
		RandStdShiftIntensityd(
			keys=["image"],
			prob=0.50,
			factors=(-10,10),
		),
		RandHistogramShiftd(
			keys=["image"],
			num_control_points=10,
			prob=0.50,
		),
		RandAdjustContrastd(
			keys=["image"],
			prob=0.50,
			gamma=(0.5,2.0),
		),
		RandZoomd(
			keys=["image","label"],
			prob=0.30,
			min_zoom=0.8,
			max_zoom=1.3,
			keep_size=False,
   			mode=[i if not i=='bilinear' else 'trilinear' for i in config["Interpolation"]], 
		),
		RandRotated(
			keys=["image","label"],
			range_x=0.5,
			range_y=0.5,
			range_z=0.5,
			prob=0.30,
			mode=config["Interpolation"], 
		),
		FgBgToIndicesd(
			keys=["label"],
		),
		RandCropByPosNegLabeld(
			keys=["image", "label"],
			label_key="label",
			spatial_size=config["INPUT_SHAPE"],
			num_samples=config["SAMPLES"],
			fg_indices_key="label_fg_indices",
			bg_indices_key="label_bg_indices",
			allow_smaller=True,
		),
		ct.SpatialPadd(    # ensure the minimum shape
			keys=["image","label"],
			spatial_size=numpy.array(config["INPUT_SHAPE"]),
			mode="minimum",
		),
		ToTensord(keys=["image", "label"]),
	]
	 
	## validation transforms
	val_transforms += [ 
		ct.SpatialPadd(    # ensure the minimum shape
			keys=["image","label"],
			spatial_size=numpy.array(config["INPUT_SHAPE"])+5,
			mode="minimum",
		),
		ToTensord(keys=["image", "label"]),
	]
	return train_transforms, val_transforms
