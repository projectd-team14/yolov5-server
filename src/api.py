import os
import pafy
import cv2
import requests
from fastapi import FastAPI
import subprocess
from subprocess import PIPE
from time import sleep
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
URL = os.environ['LARAVEL_URL']

# 検出処理を開始
@app.get("/detect/")
async def root(id: int = 0, status: int = 0):
    url = '%s/api/get_url/%s' % (URL, id)
    r = requests.get(url)
    camera_url = r.json() 
    subprocess.Popen('python ./Yolov5_DeepSort_Pytorch/main.py --save-crop --source "%s" --camera_id %s --yolo_model ./Yolov5_DeepSort_Pytorch/model_weight/best.pt' % (camera_url[0]['cameras_url'], int(id)), shell=True)
        
# ラベル付け設定
@app.get("/label/")
async def label(id: int = 0):
    url = '%s/api/get_url/%s' % (URL, id)
    r = requests.get(url)
    camera_url = r.json() 

    dir_path = './label_imgs'
    ext = 'jpg'
    
    if os.path.isfile('./label_imgs/%s.jpg' % id):
        os.remove('./label_imgs/%s.jpg' % id)

    video = pafy.new(camera_url[0]['cameras_url'])
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)

    if not cap.isOpened():
        return
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, str(id))

    ret, frame = cap.read()
    cv2.imwrite('{}.{}'.format(base_path, ext), frame)

    return FileResponse('./label_imgs/%s.jpg' % id)

# 自転車の画像
@app.get("/bicycle/")
async def bicycle(camera_id: int = 0, bicycle_id: int = 0):
    return FileResponse('./bicycle_imgs/%s/%s.jpg' % (camera_id, bicycle_id))

# テスト
@app.get("/test/")
async def bicycle():
    url = '%s/api/get_url/100' % URL
    r = requests.get(url)
    camera_url = r.json()

    return camera_url[0]['cameras_url']
