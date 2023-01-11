# yolov5-server  
このプロジェクトはAWSのGPUインスタンスに設置する機械学習用インスタンスです。画像出力と機械学習用インスタンスにコマンドを送るためにFastAPIが設置されています。  
## 環境構築
※このプロジェクトは[駐輪場管理業務支援システム](https://github.com/projectd-team14/bicycle-system)の環境構築が必要です。  
  
1.プロジェクトのclone  
```
git clone https://github.com/projectd-team14/yolov5-server.git
```
2.ディレクトリに移動してビルド  
```
cd yolov5-sever  
docker compose up -d --build
docker compose exec python3 sh
pip install -r requirements.txt
```
3.FastAPIのサーバーを立ち上げる
```
uvicorn api:app --host=0.0.0.0 --port=9000
```
