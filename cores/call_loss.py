import sys
import torch.nn as nn
from monai.losses import *
# from core.loss.Portion_losses import DiceCELoss_Portion, DiceFocalLoss_Portion
# from core.loss.Hausdorff import HausdorffBinaryLoss, DistanceMapBinaryDiceLoss
# from core.loss.clDice import clDice
# from core.loss.ActiveContour import ACLoss3DV2, FastACELoss3D # ACLoss3D, ACLoss3DV2, ACELoss3D, FastACELoss3D, FastACELoss3DV2
# from core.loss.Confusion import RecallWeightedCELoss, FalseWeightedCELoss
# from core.loss.NaviLoss import NaviLoss_B, NaviLoss_M
# from core.loss.MaskedLoss import MaskedLoss
# from core.loss.common import SoftDiceLoss
from Portion_losses import DiceCELoss_Portion, DiceFocalLoss_Portion
from Hausdorff import HausdorffBinaryLoss, DistanceMapBinaryDiceLoss
from clDice import clDice
from ActiveContour import ACLoss3DV2, FastACELoss3D # ACLoss3D, ACLoss3DV2, ACELoss3D, FastACELoss3D, FastACELoss3DV2
from Confusion import RecallWeightedCELoss, FalseWeightedCELoss
from NaviLoss import NaviLoss_B, NaviLoss_M
from MaskedLoss import MaskedLoss
from common import SoftDiceLoss


class call_loss(nn.Module):  
    def __init__(self, 
                loss_mode, 
                y_onehot=False,
                config=None):
        super().__init__()
        
        self.config = config
        sigmoid=False
        softmax=False
        
        if config["CHANNEL_OUT"] == len(config["CLASS_NAMES"].keys()):
            include_background=True
        else:
            include_background=False
        
        if '+' in loss_mode:
            loss_names = loss_mode.split('+')
        else:
            loss_names = [loss_mode]

        self.losses = []; self.with_volumes = []; self.with_classes = []
        for loss_name in loss_names:
            if loss_name=='dice':
                # self.losses.append(DiceLoss(
                #     include_background=include_background, 
                #     sigmoid=sigmoid, 
                #     softmax=softmax, 
                #     to_onehot_y=y_onehot
                # ))
                self.losses.append(SoftDiceLoss(
                    include_background=include_background, 
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='gdice':
                self.losses.append(GeneralizedDiceLoss(
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                    to_onehot_y=y_onehot
                    ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='dice_ce':
                self.losses.append(DiceCELoss(
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                    to_onehot_y=y_onehot
                    ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='dice_focal':
                self.losses.append(DiceFocalLoss(
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                    to_onehot_y=y_onehot
                    ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='dice_ce_portion':
                self.losses.append(DiceCELoss_Portion(
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                    to_onehot_y=y_onehot
                    ))
                self.with_volumes.append(True); self.with_classes.append(False)
            elif loss_name=='dice_focal_portion':
                self.losses.append(DiceFocalLoss_Portion(
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax, 
                    to_onehot_y=y_onehot
                ))
                self.with_volumes.append(True); self.with_classes.append(False)
            elif loss_name=='cl':
                self.losses.append(clDice(
                    iter_=config["CL_PARAMS"]["itr"],
                    kernel_size=config["CL_PARAMS"]["kernel"],
                    stride=config["CL_PARAMS"]["stride"],
                    padding=config["CL_PARAMS"]["padding"],
                    sigmoid=sigmoid, softmax=softmax, threshold=0.0
                ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='hd':
                self.losses.append(HausdorffBinaryLoss(
                    sigmoid=sigmoid, softmax=softmax,
                    alpha=2.0, omega=256*256, threshold=0.5
                ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='dist':
                self.losses.append(DistanceMapBinaryDiceLoss(
                    threshold=0.5, smooth=1e-5, 
                    dist_wt=1.0, boundary_fill=False
                ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='navi':
                if len(list(config["CLASS_NAMES"].keys()))==1:
                    self.losses.append(NaviLoss_B(
                        sigmoid=True
                    ))
                else:
                    self.losses.append(NaviLoss_M(
                        sigmoid=True
                    ))
                self.with_volumes.append(False); self.with_classes.append(False)
            elif loss_name=='maskednavi':
                self.losses.append(MaskedLoss(
                    mode='navi',
                    sigmoid=sigmoid, 
                    workers=10,
                    # preset=config['universal_class_preset']
                ))
                self.with_volumes.append(False); self.with_classes.append(True)
            elif loss_name=='maskeddice':
                self.losses.append(MaskedLoss(
                    mode='dice',
                    include_background=include_background,
                    sigmoid=sigmoid, 
                    softmax=softmax,
                    workers=10,
                    # preset=config['universal_class_preset']
                ))
                self.with_volumes.append(False); self.with_classes.append(True)
                
        if "LOSS_WEIGHTS" not in config.keys():
            self.weights = [1.0] * len(self.losses)
        else:
            self.wegiths = config["LOSS_WEGITHS"]
        if len(self.losses)==0: 
            sys.exit('Wrong Loss!', loss_mode)

    def forward(self, pred, target, volumes=None, classes=None):
        loss_ = 0
        for loss_fnc, with_volume, with_class, weight in zip(self.losses, self.with_volumes, self.with_classes, self.weights):
            if with_volume==True and with_class==False:
                loss_ += loss_fnc(pred, target, volumes=volumes) * weight
            elif with_volume==False and with_class==True:
                loss_ += loss_fnc(pred, target, classes=classes) * weight
            elif with_volume==True and with_class==True:
                loss_ += loss_fnc(pred, target, volumes=volumes, classes=classes) * weight
            else:
                temp = loss_fnc(pred, target) * weight
                loss_ += temp

        return loss_ / sum(self.weights)