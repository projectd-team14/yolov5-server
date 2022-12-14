# yolov5-server  
LaravelのAPIを経由してDBにアクセスするためのAPIサーバー。  
※このプロジェクトは以下のプロジェクトの環境構築が必要です。  
・[bicycle-system(駐輪場管理システム)](https://github.com/projectd-team14/bicycle_system)
## 概要  
このプロジェクトはAWSのGPUインスタンスに設置するYOLOv5専用のAPIサーバーです。FastAPIで起動しておりYOLOv5から得られた情報をLaravelのAPIサーバーに送ります。このサーバーはPublic状態になっているためSQLの直書きやセキュリティ対策が必要なデータの保存はできません。
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
srcディレクトリに入り初回のみライブラリのインストールを行う。FastAPIを立ち上げる
```
cd src
pip install -r requirements.txt
uvicorn api:app --host=0.0.0.0 --port=9000
```
