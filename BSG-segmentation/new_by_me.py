import json
from io import BytesIO
from PIL import Image
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torchvision import transforms

from unet import UNet
@st.cache_data()
def load_model() -> UNet:#str = 'models/trained_model_resnet50.pt'
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    # model = ResnetModel(path_to_pretrained_model=path)
    # model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)
    model = UNet(in_channels=3, num_classes=2, base_c=32)
    return model

@st.cache_data()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files

@st.cache_data()
def load_s3_file_structure(path: str = 'output.json') -> dict:  # str = 'src/all_image_files.json'
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)
weights_path = r'200epoch.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model()
# load weights
model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
model.to(device)

all_image_files = load_s3_file_structure()
index_of_image = sorted(list(all_image_files.keys()))
st.title('欢迎查看脑干胶质瘤的分割结果展示')
instructions = """
在检测脑干胶质瘤的时候，钆基造影剂对人体肾脏的损伤很大，于是我们通过利用计算机技术来代替钆基造影剂在寻找病灶位置中的作用，从而减少对人体的伤害.
下面将展示的是待检测病灶的大脑图片，通过造影剂显示的病灶，以及通过深度学习的分割模型所得出的病灶。希望能够直观的看到二者之间的差异，并且提出进一步改进。
    """
st.write(instructions)
dataset_type = st.sidebar.selectbox(
            "选择想查看的图像", index_of_image)
image_files_subset = all_image_files[dataset_type]['data_path']
label_files_subset = all_image_files[dataset_type]['label_path']

mean = (0.709, 0.381, 0.224)
std = (0.127, 0.079, 0.043)
print('okkkkkk')


print('ooood',image_files_subset)

image = Image.open(image_files_subset)
# from pil image to tensor and normalize
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)])
img = data_transform(image)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
output = model(img.to(device))
prediction = output['out'].argmax(1).squeeze(0)
prediction = prediction.to("cpu").numpy().astype(np.uint8)
# 将前景对应的像素值改成255(白色)
prediction[prediction == 1] = 255
# 将不敢兴趣的区域像素设置成0(黑色)
# prediction[roi_img == 0] = 0
mask = Image.fromarray(prediction)
resized_mask = mask.resize((240,240))

resized_image = image.resize((240, 240))

label = Image.open(label_files_subset)
resized_label = label.resize((240,240))

col1, col2,col3 = st.columns(3)
with col1:
    st.image(resized_image, caption='大脑成像', use_column_width='auto')
with col2:
    st.image(resized_label, caption='病灶位置')
with col3:
    st.image(resized_mask, caption='segmented_by_UNet_200epoch')

