from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
import numpy as np
from scipy import spatial
import os
import sys
import math
import shutil

def transform(img_org):
    preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    return preprocess(img_org)


path=sys.argv[1]
cap = cv2.VideoCapture(path) #ビデオ読み込み

#画像の特徴量をVGG16で抽出
vgg16 = models.vgg16(pretrained=True)
vgg16.avgpool = torch.nn.Identity()
vgg16.classifier = torch.nn.Identity()
vgg16.eval()

#フレームサイズを取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#フレームレート取得
fps = cap.get(cv2.CAP_PROP_FPS)

#フレーム数
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

key_vec=[]
data=[]
key_points=[]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n=10

#動画を読み込んで、各フレームを特徴量に変換、１つ前のフレームとの類似度を測定
#閾値で現在のフレームが１つ前のフレームのシーンの延長かを判定する
for idx in tqdm(range(int(count))):
    ret, img = cap.read()
    img=transform(Image.fromarray(img))
    img=img[None]
    result = vgg16(img)
    vec=result[0].to(device).detach().numpy().copy()
    if idx==0:
        pre_vec=vec
    else:
        result = 1 - spatial.distance.cosine(vec, pre_vec)
        if result<0.4:
            key_points.append(idx)
            key_vec.append(pre_vec)
            key_vec.append(vec)
        pre_vec=vec

key_frame=[key_points[0]]
pre=key_points[0]
for point in key_points[1:]:
    key_frame.append(point-pre)
    pre=point
key_frame.append(int(count)-sum(key_frame))
print(key_frame)
for m in range(10):
    for n,i in enumerate(key_frame):
        if i<m:
            if n>0:
                before=1 - spatial.distance.cosine(key_vec[2*n-1], key_vec[2*n-2])
                after=1 - spatial.distance.cosine(key_vec[2*n], key_vec[2*n+1])
    
                if before>after:
                    #前のシーンと結合
                    key_frame[n-1]+=key_frame[n]
                    key_frame.pop(n)
                else:
                    key_frame[n+1]+=key_frame[n]
                    key_frame.pop(n)
                    
print(key_frame)    

#上記までで作成した分割点情報を使用して、再度読み込んだ動画を分割する
cap = cv2.VideoCapture(path)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out="output/"

c=0

for n,i in enumerate(tqdm(key_frame)):
    writer = cv2.VideoWriter(out+'{}.mp4'.format(c), fmt, fps, (width, height))
    for _ in range(i):
        ret, img = cap.read()
        writer.write(img)
    writer.release()
    c+=1
            
cap.release()
cv2.destroyAllWindows()
    