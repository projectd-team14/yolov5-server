# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math
import sys
sys.path.insert(0, './yolov5')

import boto3
import requests
import json
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
from matplotlib import path
import datetime

from decimal import Decimal
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import shutil
from PIL import Image 
import imagehash 
from dotenv import load_dotenv
load_dotenv()

# S3(本番環境のみ使用)
BUCKET_NAME=os.environ['BUCKET_NAME']
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']
s3 = boto3.client('s3', region_name='ap-northeast-1', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)

APP_ENV = os.environ['APP_ENV']
URL = os.environ['LARAVEL_URL']
TIME_SLEEP = int(os.environ['TIME_SLEEP'])
UPDATE_CYCLE = int(os.environ['UPDATE_CYCLE'])
MAINTENANCE_COUNT = int(os.environ['MAINTENANCE_COUNT'])

# サーバーの状態を判定
def get_server(camera_id):
    url = '%s/api/server_condition/%s' % (URL, camera_id)
    r = requests.get(url)
    server_condition = r.json() 
    server_condition = server_condition['condition']

    return server_condition

# メンテナンス後の処理(トリミング画像の類似度を比較してYOLOv5用のIDを更新する)
def fix(camera_id):
    bicycle_ex_lis = []
    fix_path = "./bicycle_imgs_fix/%s" % camera_id
    fix_files = os.listdir(fix_path)
    old_path = "./bicycle_imgs/%s" % camera_id
    old_files = os.listdir(old_path)

    for i in range(len(fix_files)):
        for i2 in range(len(old_files)):
            hash = imagehash.average_hash(Image.open('./bicycle_imgs_fix/%s/%s' % (camera_id, fix_files[i]))) 
            otherhash = imagehash.average_hash(Image.open('./bicycle_imgs/%s/%s' % (camera_id, old_files[i2])))
            threshold = hash - otherhash
            if threshold <= 15:
                old = old_files[i2].replace('.jpg', '')
                new = fix_files[i].replace('jpg', '')
                update_lis = {
                    'old' : old,
                    'new' : new
                }
                bicycle_ex_lis.append(update_lis)

    url = '%s/api/server_update/%s' % (URL, camera_id)
    item_data = bicycle_ex_lis
    requests.post(url, json=item_data)
    shutil.rmtree(fix_path)
    shutil.rmtree(old_path)

# 停止ボタンによる処理
def stop(camera_id):
    url = '%s/api/get_camera_status/%s' % (URL, camera_id)
    r = requests.get(url)
    camera_status = r.json()
    if 'Stop' in camera_status[0]['cameras_status']:
        url = '%s/api/get_camera_stop/%s' % (URL, camera_id)
        r = requests.get(url)                  
        shutil.rmtree('Yolov5_DeepSort_Pytorch/runs/track/')
        shutil.rmtree('./bicycle_imgs/%s' % camera_id)
        if not APP_ENV == 'local':
            s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_ACCESS_KEY)
            bucket = s3.Bucket(BUCKET_NAME)
            bucket.objects.filter(Prefix='bicycle_imgs/%s' % camera_id).delete()
        exit()

# 定期実行、DBの情報を更新するタイミングを決める
def time_cycle(count_cycle):
    if count_cycle >= UPDATE_CYCLE:
        return True
    else:
        return False

# ラベルの取得
def get_label(camera_id, labels):
    url = '%s/api/get_label/%s' % (URL, camera_id)
    r = requests.get(url)
    label_lis = r.json() 
    
    if label_lis:
        json_load = json.loads(label_lis[0]['labels_json'])    
    else:
        json_load =[{
            "label_mark" : "None",
            "label_point1X" : 0,
            "label_point1Y" : 0,
            "label_point2X" : 0,
            "label_point2Y" : 720,
            "label_point3X" : 1280,
            "label_point3Y" : 720,
            "label_point4X" : 1280,
            "label_point4Y" : 0,
        }] 
    
    for i in range(len(json_load)):
        label_ap = [json_load[i]["label_mark"],json_load[i]["label_point1X"],json_load[i]["label_point1Y"],json_load[i]["label_point2X"],json_load[i]["label_point2Y"],json_load[i]["label_point3X"],json_load[i]["label_point3Y"],json_load[i]["label_point4X"],json_load[i]["label_point4Y"]]
        labels.append(label_ap)

    return labels

# 検出範囲のポリゴン生成を生成してクエリ用データを作成
def label_polygon(id, labels, output, poly, update_cycle, bicycle_lis, spots_id, camera_id, request_lis, tracking_lis, server_condition, bboxes, imc):
    label_name = labels[poly][0]
    P1X = labels[poly][1]
    P1Y = labels[poly][2]
    P2X = labels[poly][3]
    P2Y = labels[poly][4]
    P3X = labels[poly][5]
    P3Y = labels[poly][6]
    P4X = labels[poly][7]
    P4Y = labels[poly][8]
    polygon = path.Path(
        [
            [P1X, P1Y],
            [P2X, P2Y],
            [P3X, P3Y],
            [P4X, P4Y],
        ]
    )
    id_out = int(math.floor(id))
    X_out= int(math.floor(output[0]))
    Y_out= int(math.floor(output[1]))
    XY_out = polygon.contains_point([X_out, Y_out])

    if XY_out:
        if update_cycle:
            if not id in bicycle_lis:
                item_data = {
                    "type" : "insert",
                    "spots_id" : spots_id,
                    "cameras_id" : camera_id,
                    "labels_name" : label_name, 
                    "get_id" : id_out,
                    "bicycles_x_coordinate" : X_out,
                    "bicycles_y_coordinate" : Y_out,
                }
                request_lis.append(item_data)
            elif id in bicycle_lis:
                item_data = {
                    "type" : "update",
                    "spots_id" : spots_id,
                    "cameras_id" : camera_id,
                    "labels_name" : label_name, 
                    "get_id" : id_out,
                    "bicycles_x_coordinate" : X_out,
                    "bicycles_y_coordinate" : Y_out,
                }
                request_lis.append(item_data)

        if not id in tracking_lis:
            tracking_lis.append(id)

        # 画像を保存
        if server_condition == 'true':
            file_name = 'bicycle_imgs/%s/%s.jpg' % (camera_id, int(id))
            if update_cycle:
                is_file = os.path.exists(file_name)
                if not is_file:
                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                    file_path = Path("./bicycle_imgs/") / str(camera_id) / f'{int(id)}.jpg'
                    # file_path_json = "bicycle_imgs/%s/%s" % (id_str,jpg)
                    save_one_box(bboxes, imc, file_path, BGR=True)
                    if not APP_ENV == 'local':
                        s3.upload_file(file_name, BUCKET_NAME, file_name)
                        os.remove(file_name)
        else:
            file_name = "bicycle_imgs_fix/%s/%s.jpg" % (camera_id, int(id))
            is_file = os.path.exists("./bicycle_imgs_fix/%s/%s.jpg" % (camera_id, int(id)))
            if not is_file:
                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                file_path = Path("./bicycle_imgs_fix/") / str(camera_id) / f'{int(id)}.jpg'
                # file_path_json = "bicycle_imgs_fix/%s/%s" % (id_str,jpg)
                save_one_box(bboxes, imc, file_path, BGR=True)

    return label_name, id_out, X_out, Y_out, XY_out, request_lis, tracking_lis

# クエリ作成用データを基盤サーバーに送る、違反車両の更新
def post_bicycle(camera_id, request_lis, spots_time, id_collect, violation_lis):
    url = '%s/api/bicycle_update' % URL
    item_data = request_lis
    r = requests.post(url, json=item_data)  
    try:
        response_lis = r.json()
    except Exception:
        pass

    for i in range(len(response_lis)):
        out_time = spots_time
        up = response_lis[i]['updated_at']
        cr = response_lis[i]['created_at']
        updated_at = datetime.datetime.fromisoformat(up[:-1])
        created_at = datetime.datetime.fromisoformat(cr[:-1])
        time_dif = updated_at - created_at
        time_total = time_dif.total_seconds() 
        id_collect.append(response_lis[i]['get_id'])
        
        if time_total >= out_time:
            if response_lis[i]['bicycles_status'] == "None" or response_lis[i]['bicycles_status'] == "無効":
                violation_lis.append(response_lis[i]['get_id'])

    url = '%s/api/bicycle_violation' % URL
    item_data = {
        "camera_id" : camera_id,
        "violation_list" : violation_lis
    }
    r = requests.post(url, json=item_data)

    return id_collect

# トラッキングデータをもとに定期更新        
def tracking_update(id_all_lis, count_cycle, tracking_average_lis):
    for i in range(count_cycle):
        for i2 in range(len(tracking_average_lis[i])):
            if not [tracking_average_lis[i][i2], 0] in id_all_lis:
                id_all_lis.append([tracking_average_lis[i][i2], 0])

    for i3 in range(len(id_all_lis)):
        count_tracking = sum(tracking_average_lis, []).count(id_all_lis[i3][0])
        if (count_cycle / 2) >= count_tracking:
            id_all_lis[i3][1] = 1
            
    return id_all_lis

# 自転車の削除
def post_delete(camera_id, id_all_lis):
    delete_lis = []
    for i in range(len(id_all_lis)):
        if id_all_lis[i][1] == 1:
            delete_lis.append(id_all_lis[i][0])
            trimming_path = "./bicycle_imgs/%s/%s.jpg" % (camera_id, int(id_all_lis[i][0]))
            if os.path.exists(str(trimming_path)):
                os.remove(trimming_path)
                delete_file_path = os.path.join('bicycle_imgs/%s' % camera_id, '%s.jpg' % int(id_all_lis[i][0]))
                if not APP_ENV == 'local':
                    s3.delete_object(Bucket=BUCKET_NAME, Key=delete_file_path)

    url = '%s/api/bicycle_delete/%s' % (URL, camera_id)
    item_data = {
        "delete_list" : delete_lis
    }
    requests.post(url, json=item_data)

# 検出
def detect(opt):
    parser = argparse.ArgumentParser()  
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--camera_id', type=str, default=0)
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')

    args = parser.parse_args()

    camera_id = args.camera_id
    camera_id = int(camera_id)

    # 計測時間記録用
    spots_id = 0
    spots_time = 0
    update_cycle = False
    count_cycle = 0
    time_count = 0
    labels = []
    id_collect = []
    id_all_lis = []
    bicycle_lis = []
    tracking_lis = []
    tracking_average_lis = []
    delete = './bicycle_imgs/%s/' % camera_id

    # 設定時間の取得
    url = '%s/api/over_time/%s' % (URL, camera_id)
    r = requests.get(url)
    id_lis = r.json() 
    spots_id = id_lis[0]['spots_id']
    spots_time = id_lis[0]['spots_over_time']
    spots_status = id_lis[0]['spots_status']

    # メンテナンス後の判定
    server_condition = get_server(camera_id)

    # ラベルの取得
    get_label(camera_id, labels)

    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    save_imgs = increment_path(Path("./bicycle_imgs/"), exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path1, im, im0s, vid_cap, s) in enumerate(dataset):
        time_count = time_count + 1
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path1[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # 停止ボタンによる処理
            if server_condition == 'true':
                if update_cycle:
                    stop(camera_id)

            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path1[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path1, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                count_cycle = count_cycle + 1
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # 自転車(bicycle)
                    n_1 = (det[:, -1] == 0).sum()
                    bicycle = f"{n_1} "#{'A'}{'s' * (n_1 > 1)}, "

                    # 自動車(car)
                    n_2 = (det[:, -1] == 1).sum()
                    car = f"{n_2} "#{'A'}{'s' * (n_1 > 1)}, "

                    # バイク(motorcycles)
                    n_3 = (det[:, -1] == 2).sum()
                    motorcycles = f"{n_1} "#{'A'}{'s' * (n_1 > 1)}, "

                    count_status = 0

                    if spots_status == 1:
                        count_status = bicycle
                    elif spots_status == 2:
                        count_status = motorcycles
                    elif count_status == 3:
                        count_status = bicycle + motorcycles
                    else:
                        count_status = bicycle

                    # 自転車の混雑度を更新
                    if server_condition == 'true':
                        if update_cycle:
                            url = '%s/api/get_camera_count/%s/%s' % (URL, camera_id, count_status)
                            r = requests.get(url)

                    # 停止ボタンによる処理
                    if server_condition == 'true':
                        if update_cycle:
                            stop(camera_id)
                    
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    # 自転車の固有IDを検索
                    if server_condition == 'true':
                        if update_cycle:
                            url = '%s/api/get_id/%s' % (URL, camera_id)
                            r = requests.get(url)
                            bicycle_lis = r.json()
                            
                    violation_lis = []
                    request_lis = []
                    tracking_lis = []

                    for j, (output) in enumerate(outputs[i]):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        for poly in range(len(labels)):
                            # 検出範囲のポリゴン生成を生成してクエリ用データを作成
                            label_polygon(id, labels, output, poly, update_cycle, bicycle_lis, spots_id, camera_id, request_lis, tracking_lis, server_condition, bboxes, imc)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0] # X座標
                            bbox_top = output[1] # Y座標
                            bbox_w = output[2] - output[0] # 幅
                            bbox_h = output[3] - output[1] # 高さ
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                        # 画像のトリミング
                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{id:0.0f} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            '''
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_imgs / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)                            
                            '''

                    # クエリ作成用データを基盤サーバーに送る、違反車両の更新
                    if server_condition == 'true':
                        if update_cycle:
                            post_bicycle(camera_id, request_lis, spots_time, id_collect, violation_lis)

                tracking_average_lis.append(tracking_lis)
                if count_cycle >= UPDATE_CYCLE:
                    tracking_update(id_all_lis, count_cycle, tracking_average_lis)
                    tracking_average_lis.clear()

                # 自転車の削除    
                if server_condition == 'true':
                    if update_cycle:
                        post_delete(camera_id, id_all_lis)
                        id_all_lis.clear()

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
                # メンテナンス後の処理(トリミング画像の類似度を比較してYOLOv5用のIDを更新する)
                if server_condition == 'false':
                    time.sleep(TIME_SLEEP)
                    if time_count >= MAINTENANCE_COUNT:  
                        server_condition = 'true'
                        fix(camera_id)
                
                # DBの更新タイミングを調整、戻り値がTrueの場合に基盤サーバーにリクエストを送る
                update_cycle = time_cycle(count_cycle)
                if update_cycle:
                    count_cycle = 0

                bicycle_lis.clear()
                if server_condition == 'true':
                    id_collect.clear()
                
                time.sleep(5)

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=str, default=0)
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    #parser.add_argument('--spot_id', type=int, default=0)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)

