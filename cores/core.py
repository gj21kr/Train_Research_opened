import os, sys, glob, gc
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm

from monai.data import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference
from monai.transforms import RandSpatialCropSamplesd

# try:
# 	from core.valid_utils import *   
# 	from core.loss.SDF import SDFLoss
# except:
from valid_utils import *   
from loss.SDF import SDFLoss
    
class trainer:
	def __init__(self, config, logging):
		# Creates once at the beginning of training
		if config["AMP"] == True:
			self.amp = True
			self.scaler = torch.cuda.amp.GradScaler(enabled=True)
			self.type_ = torch.float16
		else:
			self.amp = False
			self.type_ = torch.float32

		try:
			if config["ACTIVATION"].lower()=='sigmoid':
				self.activation = torch.nn.Sigmoid()
			elif config["ACTIVATION"].lower()=='softmax':
				self.activation = torch.nn.Softmax()
		except:
			print('Set Activation Type on Configuration. config["Activation"]=?')
			self.activation = torch.nn.Sigmoid() # softmax : odd result! ToDO : check!

		if config["CHANNEL_OUT"] == len(config["CLASS_NAMES"].keys()):
			self.inc_back = True
		else:
			self.inc_back = False

		if "SDF" in list(config.keys()) and config["SDF"][0]==True:
			self.sdf = True 
			self.sdf_weight = config["SDF"][1]
			self.sdf_loss_function = SDFLoss(calc_sdf=True)
		else:
			self.sdf = False

		self.logging = logging
		self.log_dir = config["LOGDIR"]
		self.save_freq = config["SAVE_FREQUENT"]
		self.threshold = config["THRESHOLD"]
		self.input_shape = config["INPUT_SHAPE"]
		self.deep_supervision = config["DEEP_SUPERVISION"]
		self.num_classes = len(config["CLASS_NAMES"].keys())
		self.class_names = config["CLASS_NAMES"]
		self.visual_axis = config["VISUAL_AXIS"]

	def validation(self, valid_loader_): 
		gc.collect(); torch.cuda.empty_cache()
		self.model.eval() 
		epoch_iterator = tqdm(
		    valid_loader_, dynamic_ncols=True
		)
		dice_class, mr_class, fo_class = [], [], []
		with torch.no_grad():
			for step, batch in enumerate(epoch_iterator): # enumerate(valid_loader_):   
				val_inputs, val_labels = batch["image"], batch["label"]
				torch.cuda.empty_cache()
				if self.amp==True:             
					with torch.cuda.amp.autocast():                                
						val_outputs = sliding_window_inference(val_inputs, self.input_shape, 4, self.model, device='cpu', sw_device='cuda')
				else:
					val_outputs = sliding_window_inference(val_inputs, self.input_shape, 4, self.model, device='cpu', sw_device='cuda')
				if self.deep_supervision[0]==True: val_outputs = val_outputs[0]
				val_labels = val_labels.to(val_outputs.device)
				val_outputs = self.activation(val_outputs)
				# print('core validation2', torch.min(val_outputs), torch.max(val_outputs))
				if self.inc_back == False:
					val_outputs = val_outputs[:,1:]; val_labels = val_labels[:,1:]

				# print('core validation', batch['image_meta_dict']['filename_or_obj'])
				val_outputs[val_outputs>=self.threshold] = 1
				val_outputs[val_outputs<self.threshold] = 0
				
				if 'universal' in config["model_name"]:
					val_labels = to_onehot_universal(val_outputs, val_labels, config['universal_class_preset'])
				dice = dice_metric(val_outputs, val_labels)
				dice_class.append(dice) 
				confusion = confusion_matrix(val_outputs, val_labels)
				mr_class.append([
					calc_confusion_metric('fnr',confusion[i]) for i in range(self.num_classes)
				])
				fo_class.append([
					calc_confusion_metric('fpr',confusion[i]) for i in range(self.num_classes)
				])
   
		dice_val = val_logger(
				dice_class, mr_class, fo_class, 
				val_inputs, val_labels, val_outputs,
				self.class_names, self.visual_axis, self.logging
			)
		del val_inputs, val_labels, val_outputs
		gc.collect(); torch.cuda.empty_cache()
		return dice_val

	def check_and_save_model(self, dice_val, dice_val_best, global_step, patient_count, val_step=None):
		if dice_val > dice_val_best:
			dice_val_best = dice_val
			val_step = global_step
			torch.save({
				"global_step": global_step,
				"model_state_dict": self.model.state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
			}, os.path.join(self.log_dir, f"model_best.pth"))
			print(
				f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
			)
		else:
			patient_count += 1
			print(
				f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
			)

		# model save
		if global_step % self.save_freq == 0 and global_step != 0 :
			torch.save({
				"global_step": global_step,
				"model_state_dict": self.model.state_dict(),
				"optimizer_state_dict": self.optimizer.state_dict(),
			}, os.path.join(self.log_dir, "model_e{0:05d}.pth".format(global_step)))
			print(
				f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
			)
		if val_step != None:
			return dice_val_best, patient_count, val_step
		else:
			return dice_val_best, patient_count

	def train(self, x, y, volumes, classes):
		self.model.train()
		self.optimizer.zero_grad()   # clear previous gradients
		# Casts operations to mixed precision
		if self.amp==True:
			with torch.cuda.amp.autocast():
				loss, dice = self.core(x, y, volumes, classes)
			self.scaler.scale(loss).backward()   # Scales the loss, and calls backward() to create scaled gradients
			self.scaler.step(self.optimizer)     # Unscales gradients and calls or skips optimizer.step()
			self.scaler.update()                 # Updates the scale for next iteration
		else:
			loss, dice = self.core(x, y, volumes, classes)
			loss.backward()
			self.optimizer.step()
		return loss, dice
		
	def core(self, x, y, volumes=None, classes=None):
		logit_map = self.model(x)
		if self.deep_supervision[0]==True: # deep supervision and cpu loss calculation
			loss, dice = 0, 0
			weights_sum = np.sum(self.deep_supervision[1])
			for i, this_map in enumerate(logit_map):
				weight = self.deep_supervision[1][i]
				y = torch.nn.functional.interpolate(y, size=this_map.shape[2:])
				if self.sdf==True and torch.sum(y)!=0:
					l1 = self.loss_function(self.activation(this_map), y, volumes=volumes, classes=classes) * weight
					l2 = self.sdf_loss_function(self.activation(this_map), y) * weight
					if torch.isnan(l2)==True:
						loss += l1
					else:
						loss += l1 * (1-self.sdf_weight) + l2 * self.sdf_weight
				else:
					loss += self.loss_function(self.activation(this_map), y, volumes=volumes, classes=classes) * weight
				dice += 1 - self.dice_loss(self.activation(this_map), y)
			loss /= weights_sum
			dice /= weights_sum
		else:
			if self.sdf==True and torch.sum(y)!=0:
				l1 = self.loss_function(self.activation(logit_map), y, volumes=volumes, classes=classes) 
				l2 = self.sdf_loss_function(logit_map, y) 
				if torch.isnan(l2)==True:
					loss += l1
				else:
					loss += l1 * (1-self.sdf_weight) + l2 * self.sdf_weight
			else:
				loss = self.loss_function(self.activation(logit_map), y, volumes=volumes, classes=classes)
			dice = 1 - self.dice_loss(self.activation(logit_map), y)

		del logit_map, y
		gc.collect(); torch.cuda.empty_cache()
		return loss, dice
