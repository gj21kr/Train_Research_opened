import os, glob
import torch
from torch import optim
from monai.data import *

# from core.utils.CosineScheduler import CosineAnnealingWarmUpRestarts
from CosineScheduler import CosineAnnealingWarmUpRestarts

def call_model(config):
	model = None
	model_name = config["MODEL_NAME"].lower()
	if 'universal' in model_name:
		from Universal_model import Universal_model
		model = Universal_model(
			img_size = config["INPUT_SHAPE"],
			in_channels = config["CHANNEL_IN"], 
			out_channels = config["CHANNEL_OUT"], 
			backbone = model_name.split('_')[-1],
			encoding = 'word_embedding',
			pretrained = True,
		)
	elif 'navi' in model_name:
		from NaviAirway import SegAirwayModel
		model = SegAirwayModel(
			in_channels = config["CHANNEL_IN"], 
			out_channels = config["CHANNEL_OUT"], 
			layer_order = 'gcr'
		)
	elif 'trans:' in model_name :
		from TransSeg import SegmentationModel
		model_name = model_name.lower()
		model_name = model_name.split(':')[-1].split('-')
		encoder = model_name[0] #beit, swint, swint3d, dino
		decoder = model_name[1] #upernet, upernet_swint, setrpup, convtrans, unetr
		model = SegmentationModel(
			in_channels = config["CHANNEL_IN"],
			out_channels = config["CHANNEL_OUT"],
			force_2d = config["Force_2D"],
			use_pretrained = config["Use_Pretrained"],
			bootstrap_method = config["Bootstrap"], # centering or inflate
			patch_size = config["FEATURE_SIZE"],
			img_size = config["INPUT_SHAPE"],
			hidden_size = config["Hidden_Size"],
			mlp_dim = config["MLP_Dim"],
			num_heads = config["HEADS"],
			num_layers = config["LAYERS"],
			dropout_rate = config["DROPOUT"],
			encoder = encoder, decoder = decoder,
		)
	elif model_name=='ys_unet':
		from YS_UNet_Vein import UNet
		model = UNet(
			in_channels = config["CHANNEL_IN"],
			num_classes = config["CHANNEL_OUT"],
			bilinear = False, 
			channels = config["CHANNEL_LIST"][0]
		)
	elif model_name=='er_net':
		from ER_Net import ER_Net
		model = ER_Net(
			channels = config["CHANNEL_IN"],
			classes = config["CHANNEL_OUT"]
		)
	elif model_name=='nnunet':
		from nnUNet import nnUNet
		model = nnUNet(
			in_channels = config["CHANNEL_IN"],
			num_classes = config["CHANNEL_OUT"],
			channels = config["CHANNEL_LIST"][0],
			deep_supervision = config["DEEP_SUPERVISION"]
		)
	elif model_name=='cas_net':
		from CAS_Net import CSNet3D
		model = CSNet3D(
			channels=config["CHANNEL_IN"],
			classes=config["CHANNEL_OUT"]
		)
	elif model_name=='se_unet':
		from SE_UNet import SE_UNet 
		model = SE_UNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			channels=config["CHANNEL_LIST"], #(32, 64, 128, 256, 512),
			strides=config["STRIDES"], #(2, 2, 2, 2),
			num_res_units=config["NUM_RES_UNITS"], #2,
			dropout=config["DROPOUT"]
		)
	elif model_name=='swin_unet':
		from CTN import CTN_SwinUNETR_UNet
		model = CTN_SwinUNETR_UNet(
			img_size = config["INPUT_SHAPE"],
			in_channel = config["CHANNEL_IN"],
			out_channel = config["CHANNEL_OUT"],
			feature_size = config["FEATURE_SIZE"],
			channels=(32, 64, 128, 256, 512),
			dropout_rate = config["DROPOUT"],
		)
	elif model_name=='unest':
		from UNEST import UNesT
		model = UNesT(
			in_channels = config["CHANNEL_IN"],
			out_channels = config["CHANNEL_OUT"],
			img_size = config["INPUT_SHAPE"],
			feature_size = config["FEATURE_SIZE"],
			patch_size = config["PATCH_SIZE"],
			dropout_rate = config["DROPOUT"],
		)
		pretrained_pth = os.path.join('/nas3/jepark/pretrained', 'unest_renal_model.pt')
		vit_weights = torch.load(pretrained_pth)["model"]
		# print(vit_weights.keys())
		# raise True

		model_dict = model.nestViT.state_dict()
		vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
		model_dict.update(vit_weights)
		model.nestViT.load_state_dict(model_dict)
		model.nestViT.requires_grad = False
		del model_dict, vit_weights
		print('Pretrained Weights Succesfully Loaded !')
	elif model_name=='unetr':
		from monai.networks.nets import UNETR
		model = UNETR(
			in_channels = config["CHANNEL_IN"],
			out_channels = config["CHANNEL_OUT"],
			img_size = config["INPUT_SHAPE"],
			feature_size = config["PATCH_SIZE"],
			hidden_size = config["EMBED_DIM"],
			mlp_dim = config["MLP_DIM"],
			num_heads = config["NUM_HEADS"],
			dropout_rate = config["DROPOUT"],
			pos_embed="perceptron",
			norm_name="instance",
			res_block=True,
		)
	elif model_name == 'vnet':
		from monai.networks.nets import VNet
		model = VNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
		)
	elif model_name=='unet':
		from monai.networks.nets import UNet
		model = UNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			channels=config["CHANNEL_LIST"], #(32, 64, 128, 256, 512),
			strides=config["STRIDES"], #(2, 2, 2, 2),
			num_res_units=config["NUM_RES_UNITS"], #2,
		)
	elif model_name=='dynunet':
		from monai.networks.nets import DynUNet
		assert config["DynUnet_strides"][0] == 1, "Strides should be start with 1"
		model = DynUNet(
			spatial_dims=3,
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			kernel_size=config["DynUnet_kernel"],#[3,3,3,3,3],
			strides=config["DynUnet_strides"],#[1,2,2,2,2],
			upsample_kernel_size=config["DynUnet_upsample"],#[2,2,2,2,2],
			filters=config["DynUnet_filters"],#[64, 96, 128, 192, 256, 384, 512, 768, 1024],
			dropout=config["DROPOUT"],
			deep_supervision=True,
			deep_supr_num=len(config["DynUnet_strides"])-2,
			norm_name='INSTANCE',
			act_name='leakyrelu',
			res_block=config["DynUnet_residual"], #False
			trans_bias=False,
		)
	elif model_name=='swin_unetr':
		from monai.networks.nets import SwinUNETR
		model = SwinUNETR(
			img_size=config["INPUT_SHAPE"],
			in_channels=config["CHANNEL_IN"],
			out_channels=config["CHANNEL_OUT"],
			feature_size=config["FEATURE_SIZE"],
			use_checkpoint=True,
		)
		pretrained_pth = os.path.join('/nas3/jepark/pretrained/swinUNETR', 'fold0.pt')
		vit_weights = torch.load(pretrained_pth)["state_dict"]

		model_dict = model.swinViT.state_dict()
		vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
		model_dict.update(vit_weights)
		model.swinViT.load_state_dict(model_dict)
		model.swinViT.requires_grad = False
		del model_dict, vit_weights
		print('Pretrained Weights Succesfully Loaded !')

	elif model_name=='dense_unet':
		from DenseUNet import DenseUNet3d
		model = DenseUNet3d(
			channels_in=config["CHANNEL_IN"],
			channels_out=config["CHANNEL_OUT"]
		)

	elif model_name=='cadd_unet':
		from CADD_UNet import CADD_UNet
		model = CADD_UNet(
			img_size=config["INPUT_SHAPE"],
			in_channel=config["CHANNEL_IN"],
			out_channel=config["CHANNEL_OUT"],
			channel_list=config["CHANNEL_LIST"], #[32, 64, 128, 256, 512],
			kernel_size=config["KERNEL_SIZE"],
			drop_rate=config["DROPOUT"],
			padding=config["PADDING"],
			dilation=config["DILATION"],
			# deep_supervision=config["DEEP_SUPERVISION"],
		)
		if config["CHECK_MODEL"]:
			model.check_input_size()
	elif model_name=='dd_unet':
		from DD_UNet import DD_UNet
		model = DD_UNet(
			in_channel=config["CHANNEL_IN"],
			out_channel=config["CHANNEL_OUT"],
			channel_list=config["CHANNEL_LIST"], #[32, 64, 128, 256, 512],
			kernel_size=(3, 3, 3),
			drop_rate=config["DROPOUT"],
		)
	elif model_name=='voxel_resnet':
		from VoxResNet import VoxResNet
		model = VoxResNet(
			in_channels=config["CHANNEL_IN"],
			num_class=config["CHANNEL_OUT"],
		)
	elif model_name=='r2att_unet':
		from R2Att_UNet import R2AttU_Net
		model = R2AttU_Net(
			img_ch=config["CHANNEL_IN"],
			output_ch=config["CHANNEL_OUT"],
		)
	elif model_name=='deeplabv3':
		from DeeplabV3 import DeepLabV3_3D
		model = DeepLabV3_3D(
			num_classes=config["CHANNEL_OUT"],
			input_channels=config["CHANNEL_IN"],
			resnet='resnet152_os16', #resnet18_os16, resnet34_os16, resnet50_os16, resnet101_os16, resnet152_os16, 
		)
	assert model is not None, 'Model Error!'    
	return model


def call_optimizer(config, model):
	def call_function(config, model):
		if config["OPTIM_NAME"] in ['SGD', 'sgd']:
			optimizer= optim.SGD(model.parameters(), lr=config["LR_INIT"], momentum=config["MOMENTUM"])
		elif config["OPTIM_NAME"] in ['ADAM', 'adam', 'Adam']:
			optimizer= optim.Adam(model.parameters())
		elif config["OPTIM_NAME"] in ['ADAMW', 'adamw', 'AdamW', 'Adamw']:
			optimizer= optim.AdamW(model.parameters())
		elif config["OPTIM_NAME"] in ['ADAGRAD', 'adagrad', 'AdaGrad']:
			optimizer= optim.Adagrad(model.parameters(), lr=config["LR_INIT"], lr_decay=config["LR_DECAY"])
		else:
			raise Exception("Wrong Name for Optimizer!")
		return optimizer

	if config["SCHEDULER"] in ["cosine", "cosin", "cos"]:
		config["LR_INIT"] = 0
		optimizer = call_function(config, model)
		scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.01,  T_up=10, gamma=0.5)
	elif config["SCHEDULER"] in ["exp", "exponential"]:
		optimizer = call_function(config, model)
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	elif config["SCHEDULER"] in ["plateau"]:
		optimizer = call_function(config, model)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=5e-7, verbose=True)
	else:
		optimizer = call_function(config, model)
		scheduler = None
	return optimizer, scheduler