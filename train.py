import os, gc, sys
import importlib
import random
import torch
import wandb
import numpy as np
import argparse as ap
import torch.nn as nn
from tqdm import tqdm
from shutil import copyfile

import warnings
warnings.filterwarnings("ignore")

for r, d, f in os.walk(os.getcwd()):
    if os.path.isdir(r) and r not in sys.path:
        if 'cache' in r : continue 
        if 'git' in r : continue 
        if 'wandb' in r : continue
        if 'checkpoints' in r : continue         
        sys.path.insert(0, r)
        
from call_data import call_dataloader_test
from call_data import call_fold_dataset, call_dataloader, call_dataloader_monai, call_nifti
from call_model import call_model, call_optimizer
from call_loss import call_loss
from core import trainer as trainer_class
from call import call_trans_function
import decathlon_datalist as dd


def initialize(config, logging):
	model = call_model(config)
	model = nn.DataParallel(model)

	if logging:
		wandb.watch(model, log="all")
	model.to(config["DEVICE"])

	optimizer, scheduler = call_optimizer(config, model)

	if config["LOAD_MODEL"]:
		# check_point = torch.load(os.path.join(config["LOGDIR"], f"model_best.pth"))
		check_point = torch.load(config["LOAD_MODEL"])
		if 'resume_key' in config.keys():
			model_key = config['resume_key']
		else:
			model_key = 'model_state_dict'
		try:
			model.load_state_dict(check_point[model_key])
		except:
			model.load_state_dict(check_point)
		try:
			optimizer.load_state_dict(check_point['optimizer_state_dict'])
		except:
			pass
	return model, optimizer, scheduler

def main(config, logging=False):
	if logging:
		run = wandb.init(project=config["PROJ_NAME"], entity=config["ENTITY"]) 
		wandb.config.update(config) 

	# Initialize model, optimizer, loss functions
	trainer = trainer_class(config, logging)
	trainer.model, trainer.optimizer, scheduler = initialize(config, logging)
	trainer.loss_function = call_loss(loss_mode=config["LOSS_NAME"], config=config)
	trainer.dice_loss = call_loss(loss_mode='dice', config=config)

	# Dataset
	train_transforms, valid_transforms = call_trans_function(config)

	file_list = dd.load_decathlon_datalist(config["JSON"], True, 'training')
	train_list, valid_list = call_fold_dataset(file_list, target_fold=config["FOLD"], total_folds=config["FOLDS"])    

	print('Train', len(train_list), 'Valid', len(valid_list))

	if logging:
		artifact = wandb.Artifact(
			"dataset", type="dataset", 
			metadata={
				"train_list":train_list, "valid_list":valid_list, 
				"train_len":len(train_list), "valid_len":len(valid_list)
			})
		run.log_artifact(artifact)
	
	if "Array" in config["TRANSFORM"]:
		train_list = call_nifti(config, train_list)
		valid_list = call_nifti(config, valid_list)
		print('Data Loaded!')
	
	call_dataloader = call_dataloader_monai
	if 'data_loader' in config.keys():
		if config['data_loader'] == 'test':
			call_dataloader = call_dataloader_test

	# train_list = train_list[:5]; valid_list = valid_list[:3]
	
	train_loader = call_dataloader(config, train_list, train_transforms, "Train")
	valid_loader = call_dataloader(config, valid_list, valid_transforms, "Valid")

	best_loss = 1.
	dice_val_best = 0.0
	early_stop, patient_count = 5, 0  
	global_step = 0
	## Training!! 
	while global_step <= config["MAX_ITERATIONS"]:
		epoch_iterator = tqdm(
			train_loader, desc="Training (loss=X.X) (x/x)", dynamic_ncols=True
		)
		step = 0
		epoch_loss, epoch_dice = 0., 0.
		loss, dice = 0, 0; val_best_step = 0
		gc.collect(); torch.cuda.empty_cache()
		for batch in epoch_iterator:
			if global_step==0 : print("input shape is ", batch["label"].shape)
			## Validation!!
			if (global_step % config["EVAL_NUM"] == 0 and global_step != 0
			) or global_step == config["MAX_ITERATIONS"]:
				print("Validation Started!")
				dice_val = trainer.validation(valid_loader)
				dice_val_best, patient_count, val_best_step = trainer.check_and_save_model(
					dice_val, dice_val_best, global_step, patient_count, val_best_step
				)
				scheduler.step(dice_val)
				# if patient_count > early_stop:
				#     print("Early Stopped at", global_step)
				#     global_step = config["MAX_ITERATIONS"] +1
			x, y, volumes, classes = None, None, None, None
			if type(batch) == list:
				for this_batch in batch :                
					x = this_batch["image"].to(config["DEVICE"])
					y = this_batch["label"].to(config["DEVICE"])
					if "volume" in this_batch.keys(): volumes = this_batch["volume"]
					if "class" in this_batch.keys(): classes = this_batch["class"]
			else:                
				x = batch["image"].to(config["DEVICE"]) 
				y = batch["label"].to(config["DEVICE"])
				if "class" in batch.keys(): 
					classes = batch["class"]
				if "volume" in batch.keys():
					volumes = batch["volume"]
					if config["CHANNEL_OUT"]>len(config["CLASS_NAMES"].keys()):
						for b in range(len(volumes)):
							total = 0
							for c in range(volumes[0].shape[0]):
								total += volumes[b][c]
							volumes.insert(0, total)
			loss, dice = trainer.train(x, y, volumes, classes)			
			epoch_loss += loss.item()
			epoch_dice += dice.item()

			step += 1
			global_step += 1
			epoch_iterator.set_description(
				"Training (dice=%2.5f, loss=%2.5f) (%1d/%1d)" % (dice.item(), loss.item(), global_step, config["MAX_ITERATIONS"])
			)
		train_loss = epoch_loss / step
		train_dice = epoch_dice / step
		if logging:
			wandb.log({
				'train_loss': train_loss,
				'train_dice': train_dice,
			})        
		# gc.collect(); torch.cuda.empty_cache()
	if logging:
		artifact = wandb.Artifact('model', type='model')
		artifact.add_file(
			os.path.join(config["LOGDIR"], f"model_best.pth"), 
			name=f'model/{config["MODEL_NAME"]}')
		run.log_artifact(artifact)
	return True

def prepare(config):
	os.environ["CUDA_VISIBLE_DEVICES"] = config["GPUS"]
	config["GPUS"] = range(len(config["GPUS"].split(','))) #[int(g) for g in config["GPUS"].split(',')]
	if config["NUM_GPUS"] == 0:
		config["DEVICE"] = torch.device('cpu')
		gc.collect()
	else:
		config["DEVICE"] = torch.device('cuda')
		gc.collect()
		torch.cuda.empty_cache()
	torch.manual_seed(config["SEEDS"])
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.set_num_threads(config["WORKERS"])
	if torch.cuda.get_device_name(0) == 'NVIDIA A100-SXM-80GB':
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	np.random.seed(config["SEEDS"])
	return config

def save_config_file(log_dir, config_name):
	config_name += '.py'
	file_path = os.path.join(log_dir,config_name)
	copyfile(f'./config/{config_name}', file_path)

if __name__ == "__main__":
	parser = ap.ArgumentParser()
	parser.add_argument('trainer', default=None) 
	args = parser.parse_args()
	
	config = importlib.import_module(f'{args.trainer}').config
	save_config_file(config["LOGDIR"], args.trainer)

	config = prepare(config)
	main(config, logging=config["LOGGING"])