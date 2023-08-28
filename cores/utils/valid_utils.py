import torch
import wandb
import numpy as np
from numba import jit
from monai.transforms import ClassesToIndices

def val_logger(dice_class, mr_class, fo_class, val_inputs, val_labels, val_outputs, class_names, visual_axis, logging):
    dice_dict, dice_val = calc_mean_class(dice_class, 'valid_dice', class_names=class_names)
    miss_dict, miss_val = calc_mean_class(mr_class, 'valid_miss rate', class_names=class_names)
    false_dict, false_val = calc_mean_class(fo_class, 'valid_false alarm', class_names=class_names)

    if logging:
        wandb.log({
            'valid_dice': float(dice_val),
            'valid_miss rate': float(miss_val),
            'valid_false alarm': float(false_val),
            'valid_image': log_image_table(val_inputs[0].cpu(),
                                            val_labels[0].cpu(),
                                            val_outputs[0].cpu(), 
                                            class_names=class_names,
                                            visual_axis=visual_axis),
        })
        wandb.log(dice_dict)
        wandb.log(miss_dict)
        wandb.log(false_dict)
    return dice_val

def dice_calc(predicted, ground_truth, epsilon = 1e-6):
    if not torch.is_tensor(predicted): predicted = torch.Tensor(predicted)
    if not torch.is_tensor(ground_truth): ground_truth = torch.Tensor(ground_truth)
    intersection = torch.sum(predicted * ground_truth)
    union = torch.sum(predicted) + torch.sum(ground_truth)
    return (2 * intersection + epsilon) / (union + epsilon)


def dice_metric(predicted, ground_truth):
    results = []
    if predicted.ndim==3 and ground_truth.ndim==3:
        results = dice_calc(predicted, ground_truth)
    elif predicted.ndim==4 and ground_truth.ndim==4:
        for b in range(predicted.shape[0]):
            results.append(dice_calc(predicted[b],ground_truth[b]))
    elif predicted.ndim==5 and ground_truth.ndim==5:
        for b in range(predicted.shape[0]):
            for c in range(predicted.shape[1]):
                results.append(dice_calc(predicted[b,c],ground_truth[b,c]))

    results = [r for r in results if not r is None]
    return results


def conf_mat_calc(predicted, ground_truth):
    if not torch.is_tensor(predicted): input_tensor = torch.Tensor(predicted)
    if not torch.is_tensor(ground_truth): ground_truth = torch.Tensor(ground_truth)
    tn = torch.sum((predicted == 0) & (ground_truth == 0))
    fn = torch.sum((predicted == 0) & (ground_truth == 1))
    fp = torch.sum((predicted == 1) & (ground_truth == 0))
    tp = torch.sum((predicted == 1) & (ground_truth == 1))
    return [tp, fp, tn, fn]


def confusion_matrix(predicted, ground_truth):
    results = []    
    if predicted.ndim==3 and ground_truth.ndim==3:
        results = conf_mat_calc(predicted, ground_truth)
    elif predicted.ndim==4 and ground_truth.ndim==4:
        for b in range(predicted.shape[0]):
            results.append(conf_mat_calc(predicted[b],ground_truth[b]))
    elif predicted.ndim==5 and ground_truth.ndim==5:
        for b in range(predicted.shape[0]):
            for c in range(predicted.shape[1]):
                results.append(conf_mat_calc(predicted[b,c],ground_truth[b,c]))
                
    results = [r for r in results if len(r)>0]
    return results

def to_onehot(predicted, ground_truth, mode):
    if torch.is_tensor(predicted):
        lib = torch 
    else:
        lib = np 
    newGT = lib.zeros(predicted.shape)
    nClasses = predicted.shape[-4]
    for i in range(nClasses):
        if mode in [5]:
            newGT[0, i, ground_truth[0,0]==(i+1)] = 1
        elif mode in [3]:
            newGT[i, ground_truth[0]==(i+1)] = 1
        else:
            print('check data shape', predicted.shape, ground_truth, mode)
    return newGT


def label_encoder(image, num_classes, output_shape, return_inds=False):
    result = torch.zeros(output_shape)
    for this_class in range(num_classes):
        result[image[this_class]>0] = this_class + 1
    return result


def log_image_table(image, label, pred, class_names, visual_axis):
    mask_images = []; image = image[0]

    pred = label_encoder(pred, len(class_names), image.shape)
    if label.shape == pred.shape:
        label = label_encoder(label, len(class_names), image.shape)
        
    inds = np.where(label)
    if visual_axis==1:
        min_, max_ = min(inds[0]), max(inds[0])+1
    elif visual_axis==2:
        min_, max_ = min(inds[1]), max(inds[1])+1
    elif visual_axis==3:
        min_, max_ = min(inds[2]), max(inds[2])+1

    for frame in range(min_, max_,2):
        if visual_axis == 1:
            t_image = image[frame].numpy().T
            t_label = label[frame].numpy().T
            t_pred = pred[frame].numpy().T
        elif visual_axis == 2:
            t_image = image[:,frame].numpy().T
            t_label = label[:,frame].numpy().T
            t_pred = pred[:,frame].numpy().T
        elif visual_axis == 3:
            t_image = image[:,:,frame].numpy()
            t_label = label[:,:,frame].numpy()
            t_pred = pred[:,:,frame].numpy()
        t_min, t_max = np.min(t_image), np.max(t_image)
        t_image = (t_image-t_min)*255/(t_max-t_min)
        mask_images.append(wandb.Image(t_image, masks={
            "ground_truth":{"mask_data":t_label,"class_labels":class_names},
            "predictions":{"mask_data":t_pred,"class_labels":class_names},
        }))
    return mask_images



def calc_mean_class(list_, metric_class='valid_dice', class_names=[]):
    all_mean = {}; all_value = 0
    num_classes = len(list_[0])
    
    for class_ in range(num_classes):
        # mean_ = np.mean(list_[:][class_])
        mean_ = np.mean([this[class_] for this in list_])
        mesg = '{}/{}'.format(metric_class,class_names[class_+1])
        mesg = {mesg:mean_}
        all_mean.update(mesg)
        all_value += mean_

    all_value = float(all_value)/float(num_classes+1)
    return all_mean, all_value


def calc_confusion_metric(metric_name, confusion_matrix):
    tp, fp, tn, fn = confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3]
    p = tp + fn
    n = fp + tn
    # calculate metric
    metric = check_confusion_matrix_metric_name(metric_name)
    numerator: torch.Tensor
    denominator: Union[torch.Tensor, float]
    nan_tensor = torch.tensor(float("nan"))
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tn), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, nan_tensor)
        tnr = torch.where(n > 0, tn / n, nan_tensor)
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator

def check_confusion_matrix_metric_name(metric_name: str):
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    if metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    if metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    if metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    if metric_name in ["false_omission_rate", "for"]:
        return "for"
    if metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    if metric_name in ["accuracy", "acc"]:
        return "acc"
    if metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    if metric_name in ["f1_score", "f1"]:
        return "f1"
    if metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    if metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    if metric_name in ["informedness", "bookmaker_informedness", "bm"]:
        return "bm"
    if metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    raise NotImplementedError("the metric is not implemented.")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
