import os
import clip
import torch
import json

with open('/home/jepark/MIAI_Segmentation/dataset/core/public_classes.json','r') as f:
    # 46 
    # preset = [1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,55,56,57,58,82,88,89,90]
    # 44 
    preset = [1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,55,56,57,58,82,103]
    public_classes = json.load(f)
    class_names = {}
    for ind, number in enumerate(preset):
        name = public_classes[str(number)]
        if 'rib' in name : name = 'rib'
        class_names[ind] = name
ORGAN_NAME = list(class_names.keys())
num_classes = len(preset)
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, f'/home/jepark/MIAI_Segmentation/core/pretrained_weights/txt_encoding_custom_{num_classes}classes.pth')

