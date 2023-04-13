import requests
import json
import pandas as pd
import numpy as np
import math


# 서울시 열린데이터 광장 데이터
# Open API Key: 455343516c66656e39376c4341444a
# http://openapi.seoul.go.kr:8088/ [Open API Key] /json/VwsmTrdarSelngQq/1/5/20220301 + / 요청명
# 점포수 3개 미만은 마스킹 처리(*)하여 표시합니다.
columns=['기준_년_코드', '기준_분기_코드',
                           '상권_구분_코드', '상권_구분_코드_명', '상권_코드', '상권_코드_명',
                           '서비스_업종_코드', '서비스_업종_코드_명',
                           '분기당_매출_금액', '분기당_매출_건수',
                           '주중_매출_비율', '주말_매출_비율', '월요일_매출_비율', '화요일_매출_비율', '수요일_매출_비율', '목요일_매출_비율', '금요일_매출_비율', '토요일_매출_비율', '일요일_매출_비율',
                           '시간대_00~06_매출_비율', '시간대_06~11_매출_비율', '시간대_11~14_매출_비율', '시간대_14~17_매출_비율', '시간대_17~21_매출_비율', '시간대_21~24_매출_비율',
                           '남성_매출_비율', '여성_매출_비율', '연령대_10_매출_비율', '연령대_20_매출_비율', '연령대_30_매출_비율', '연령대_40_매출_비율', '연령대_50_매출_비율', '연령대_60_이상_매출_비율',
                           '주중_매출_금액', '주말_매출_금액', '월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액', '목요일_매출_금액', '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액',
                           '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액', '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
                           '남성_매출_금액', '여성_매출_금액', '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액', '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
                           '주중_매출_건수', '주말_매출_건수', '월요일_매출_건수', '화요일_매출_건수', '수요일_매출_건수', '목요일_매출_건수', '금요일_매출_건수', '토요일_매출_건수', '일요일_매출_건수',
                           '시간대_건수~06_매출_건수', '시간대_건수~11_매출_건수', '시간대_건수~14_매출_건수', '시간대_건수~17_매출_건수', '시간대_건수~21_매출_건수', '시간대_건수~24_매출_건수',
                           '남성_매출_건수', '여성_매출_건수', '연령대_10_매출_건수', '연령대_20_매출_건수', '연령대_30_매출_건수', '연령대_40_매출_건수', '연령대_50_매출_건수', '연령대_60_이상_매출_건수',
                           '점포수']

def Sell2017():

    seoulSell2017 = []

    url = 'http://openapi.seoul.go.kr:8088/455343516c66656e39376c4341444a/json/VwsmTrdarSelngQq/1/1/2017'

    response = requests.get(url).json()
    count = response['VwsmTrdarSelngQq']['list_total_count']
    print(count, type(count))

    if count >= 1000:
        max = math.ceil(count/1000)

        for i in range(max):
            if i+1 != max:
                print(str(((i+1)*1000)-999)+' 번부터',str((i+1)*1000)+' 번까지 호출')
                url = 'http://openapi.seoul.go.kr:8088/455343516c66656e39376c4341444a/json/VwsmTrdarSelngQq/'+str(((i+1)*1000)-999)+'/'+str((i+1)*1000)+'/2017'
                response = requests.get(url).text
                result = json.loads(response)
                seoulSell2017.append(result['VwsmTrdarSelngQq']['row'])
            else:
                print(str((max*1000)-999)+' 번부터',str(count)+' 번까지 호출')
                url = 'http://openapi.seoul.go.kr:8088/455343516c66656e39376c4341444a/json/VwsmTrdarSelngQq/'+str((max*1000)-999)+'/'+str(count)+'/2017'
                response = requests.get(url).text
                result = json.loads(response)
                seoulSell2017.append(result['VwsmTrdarSelngQq']['row'])

    print('처리 완료')
    df = pd.DataFrame.from_records(
        [item for sublist in seoulSell2017 for item in sublist],
        columns=columns)
    print(df.head())
    # 서울 상권 매출 csv로 저장
    df.to_csv('C:/Users/체자드/Desktop/사이드 프로젝트/Django/airproject/airproject/airapp/static/data/seoulSell2017.csv', index=False)

Sell2017()