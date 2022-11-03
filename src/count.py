from time import sleep
import requests

def main():
    # 駐輪場
    url = 'http://host.docker.internal:8000/api/get_camera_all'
    r = requests.get(url)
    camera_status = r.json()

    spots_id_lis = [] 
    for i in range(len(camera_status)):
        if not camera_status[i]['spots_id'] in spots_id_lis:
            spots_id_lis.append(camera_status[i]['spots_id'])
            print("Spot ID" + str(camera_status[i]['spots_id']))

    # 1日(1時間に1回定期処理)
    for i2 in range(len(spots_id_lis)):
        count_day1 = 0
        count_day1 = int(count_day1)
        for i3 in range(len(camera_status)):
            if camera_status[i3]['spots_id'] == spots_id_lis[i2]:
                count_day1 = count_day1 + int(camera_status[i3]['cameras_count'])

        url = 'http://host.docker.internal:8000/api/get_spot_day1/%s' % spots_id_lis[i2]
        r = requests.get(url)
        spot_day1 = r.json()

        day1 = spot_day1[0]['spots_count_day1']
        db_day_lis = day1.split(',')
        if day1 == "None":
            new_day1 = ("%s" % str(count_day1))
        else:
            new_day1 = ("%s,%s" % (day1,str(count_day1)))

        url = 'http://host.docker.internal:8000/api/get_spot_day1_update/%s' % spots_id_lis[i2]
        item_data = {
            "spots_count_day1" : new_day1
        }
        r = requests.post(url, json=item_data)

        # 25時間目の処理
        print(len(db_day_lis) + 1)
        if len(db_day_lis) >= 24:
            url = 'http://host.docker.internal:8000/api/get_spot_day1_update/%s' % spots_id_lis[i2]
            item_data = {
                "spots_count_day1" : str(count_day1)
            }
            r = requests.post(url, json=item_data)
            
            day_ave = sum([int(s) for s in db_day_lis])/len(db_day_lis)
            print("day1を更新")
            # 1週間(1日の平均を7日間)
            url = 'http://host.docker.internal:8000/api/get_spot_week1/%s' % spots_id_lis[i2]
            r = requests.get(url)
            spot_week1 = r.json()

            db_week1 = spot_week1[0]['spots_count_week1']
            db_week_lis = db_week1.split(',')
            if db_week1 == "None":
                new_week1 = ("%s" % str(day_ave))
            else:
                new_week1 = ("%s,%s" % (db_week1,str(day_ave)))

            url = 'http://host.docker.internal:8000/api/get_spot_week1_update/%s' % spots_id_lis[i2]
            item_data = {
                "spots_count_week1" : new_week1
            }
            r = requests.post(url, json=item_data)
            
            if len(db_week_lis) >= 7:
                db_week_lis.pop(0)
                db_week_lis.append(str(day_ave))
                new_week_lis = ",".join(db_week_lis)
                new_week1 = ("%s" % (new_week_lis))
                url = 'http://host.docker.internal:8000/api/get_spot_week1_update/%s' % spots_id_lis[i2]
                item_data = {
                    "spots_count_week1" : new_week1
                }
                r = requests.post(url, json=item_data)
                print("weekを更新") 

            # 1か月(30日間で固定)
            url = 'http://host.docker.internal:8000/api/get_spot_month1/%s' % spots_id_lis[i2]
            r = requests.get(url)
            spot_month1 = r.json()

            db_month1 = spot_month1[0]['spots_count_month1']
            db_month1_lis = db_month1.split(',')
            if db_month1 == "None":
                new_month1 = ("%s" % str(day_ave))
            else:
                new_month1 = ("%s,%s" % (db_month1,str(day_ave)))

            url = 'http://host.docker.internal:8000/api/get_spot_month1_update/%s' % spots_id_lis[i2]
            item_data = {
                "spots_count_month1" : new_month1
            }
            r = requests.post(url, json=item_data)

            # 30日に１回更新
            if len(db_month1_lis) >= 30:
                db_month1_lis.pop(0)
                db_month1_lis.append(str(day_ave))
                new_month1_lis = ",".join(db_month1_lis)
                new_month1 = ("%s" % (new_month1_lis))
                url = 'http://host.docker.internal:8000/api/get_spot_month1_update/%s' % spots_id_lis[i2]
                item_data = {
                    "spots_count_month1" : new_month1
                }
                r = requests.post(url, json=item_data)
                print("month1を更新")

            # 3か月(90日間で固定)
            url = 'http://host.docker.internal:8000/api/get_spot_month3/%s' % spots_id_lis[i2]
            r = requests.get(url)
            spot_month3 = r.json()

            db_month3 = spot_month3[0]['spots_count_month3']
            db_month3_lis = db_month3.split(',')
            if db_month3 == "None":
                new_month3 = ("%s" % str(day_ave))
            else:
                new_month3 = ("%s,%s" % (db_month3,str(day_ave)))
            url = 'http://host.docker.internal:8000/api/get_spot_month3_update/%s' % spots_id_lis[i2]
            item_data = {
                "spots_count_month3" : new_month3
            }
            r = requests.post(url, json=item_data)
                
            # 90日に１回更新
            if len(db_month3_lis) >= 90:
                db_month3_lis.pop(0)
                db_month3_lis.append(str(day_ave))
                new_month3_lis = ",".join(db_month3_lis)
                new_month3 = ("%s" % (new_month3_lis))

                url = 'http://host.docker.internal:8000/api/get_spot_month3_update/%s' % spots_id_lis[i2]
                item_data = {
                    "spots_count_month3" : new_month3
                }
                r = requests.post(url, json=item_data)          
                print("month3を更新")

# 定期実行
def time_cycle():
    # sleepで定期実行、デプロイ時に定期実行用プラグインに移す。
    while True:
        main()       
        print("update")
        #sleep(3600)

time_cycle()