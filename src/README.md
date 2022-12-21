# yolov5-server  
YOLOv5の推論結果をBDに送るためのAPIサーバー。  
## 概要  
このリポジトリはHerokuのインスタンスに設置するYOLOv5専用のAPIサーバーです。FastAPIで起動しておりYOLOv5から得られた情報をLaravelのAPIサーバーに送ります。
## 環境構築
プロジェクトのclone  
```
git clone https://github.com/projectd-team14/yolov5-server.git
```
ディレクトリに移動してビルド  
```
cd yolov5-sever  
docker compose up -d --build
```
コンテナ内に入る
```
docker container exec -it yolov5-server-python3-1 bash
```
srcディレクトリに入りFastAPIを立ち上げる
```
cd src
uvicorn api:app --host=0.0.0.0 --port=9000
```
