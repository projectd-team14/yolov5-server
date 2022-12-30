import os
import time
import pafy
import cv2
import requests
import subprocess
from subprocess import PIPE
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
import boto3
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
URL = os.environ['LARAVEL_URL']
APP_ENV = os.environ['APP_ENV']

# 本番環境のみ使用
BUCKET_NAME=os.environ['BUCKET_NAME']
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']
INSTAMCE_ID = os.environ['INSTAMCE_ID']

s3 = boto3.client('s3', region_name='ap-northeast-1', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
ssm = boto3.client('ssm', region_name='ap-northeast-1',aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)

# 検出処理を開始
@app.get("/detect/")
async def root(id: int = 0, status: int = 0):
    url = '%s/api/get_url/%s' % (URL, id)
    r = requests.get(url)
    camera_url = r.json() 
    detect_command = 'python ./Yolov5_DeepSort_Pytorch/main.py --save-crop --source "%s" --camera_id %s --yolo_model ./Yolov5_DeepSort_Pytorch/model_weight/dataset_all_01/best.pt' % (camera_url[0]['cameras_url'], id)
    
    # ローカル環境と本番環境で実行する内容を変える（ローカル環境：Yolov5_DeepSort_Pytorch, 本番環境：AWS EC2 g4dn.xlarge）
    if APP_ENV == 'local':
        subprocess.Popen(detect_command, shell=True)
    else:
        ssm_command(ssm, INSTAMCE_ID, ["cd /home/ec2-user", detect_command])

# ラベル付け設定
@app.get("/label/")
async def label(id: int = 0):
    url = '%s/api/get_url/%s' % (URL, id)
    r = requests.get(url)
    camera_url = r.json() 
    dir_path = 'label_imgs'
    ext = 'jpg'

    if APP_ENV == 'local':
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
    else:
        video = pafy.new(camera_url[0]['cameras_url'])
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)

        if not cap.isOpened():
            return
        
        os.makedirs(dir_path, exist_ok=True)
        base_path = os.path.join(dir_path, str(id))
        ret, frame = cap.read()
        cv2.imwrite('{}.{}'.format(base_path, ext), frame)
        file_name = 'label_imgs/%s.jpg' % id
        s3.upload_file(file_name, BUCKET_NAME, file_name)
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_name)
        body = response['Body'].read()
        os.remove(file_name)
        
        return Response(content=body, media_type="jpg")

# 自転車の画像
@app.get("/bicycle/")
async def bicycle(camera_id: int = 0, bicycle_id: int = 0):
    if APP_ENV == 'local':
        return FileResponse('./bicycle_imgs/%s/%s.jpg' % (camera_id, bicycle_id))
    else:
        file_name = 'bicycle_imgs/%s/%s.jpg' % (camera_id, bicycle_id)
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_name)
        body = response['Body'].read()

        return Response(content=body, media_type="jpg")

# GPUインスタンスにRunコマンドを送る
def ssm_command(ssm,instance, commands):
    r = ssm.send_command(
        InstanceIds = [instance],
        DocumentName = "AWS-RunShellScript",
        Parameters = {
          "commands": commands
        }
    )
    command_id = r['Command']['CommandId']

    while True:
        time.sleep(1)
        res = ssm.list_command_invocations(CommandId=command_id)
        invocations = res['CommandInvocations']

        if len(invocations) <= 0: continue
        
        status = invocations[0]['Status']
        
        if status == 'Success': 
            return True
        if status == 'Failed': 
            return False
        