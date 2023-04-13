# -*- coding: utf-8 -*-

from django.http import JsonResponse
from django.shortcuts import render
import statsmodels.formula.api as smf
import scipy.stats as stats
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings(action='ignore')  # 경고출력안하기
# from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)




# static 폴더에서 경로 불러오기
gol_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static', 'analysis_data', 'data', '용산구 골목상권(거리두기 추가).csv')
bal_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static', 'analysis_data', 'data', '용산구 발달상권(거리두기 추가).csv')
jeon_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static', 'analysis_data', 'data', '용산구 전통시장(거리두기 추가).csv')
cul_path = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static', 'analysis_data', 'data', '용산구 관광특구(거리두기 추가).csv')

gol_df = pd.read_csv(gol_path)
bal_df = pd.read_csv(bal_path)
jeon_df = pd.read_csv(jeon_path)
cul_df = pd.read_csv(cul_path)

# 19~21년 용산구 매출 데이터 통합본
yongsan_data = pd.concat([gol_df, bal_df, jeon_df, cul_df], axis=0)

def map(request):

    return render(request, 'analysis/map.html')

def calldbFunc(request):

    if request.method == "POST":

        # 19~21년 용산구 매출 데이터 통합본
        global yongsan_data
        
        BigTradingArea = request.POST.get('BigTradingArea')
        tradingArea = request.POST.get('tradingArea')
        businessType = request.POST.get('businessType')
        smallBusiType = request.POST.get('smallBusiType')

        area_data = yongsan_data[yongsan_data['상권_구분_코드_명'] == BigTradingArea]

        # 상권별 변수 및 모델 설정
        area_models = {

            '골목상권': {
                'variables_func': gol_variables,
                'formula': '분기당_매출_금액 ~ 월요일_매출_금액 + 금요일_매출_금액 + 남성_매출_금액 + 수요일_매출_금액 + 화요일_매출_금액'
            },

            '발달상권': {
                'variables_func': bal_variables,
                'formula': '분기당_매출_금액 ~ 시간대_14_17_매출_금액 + 수요일_매출_금액 + 시간대_11_14_매출_금액 + 월요일_매출_금액 + 금요일_매출_금액'
            },

            '전통시장': {
                'variables_func': jeon_variables,
                'formula': '분기당_매출_금액 ~ 금요일_매출_금액 + 수요일_매출_금액 + 여성_매출_금액 + 화요일_매출_금액 + 목요일_매출_금액'
            },

            '관광특구': {
                'variables_func': cul_variables,
                'formula': '분기당_매출_금액 ~ 남성_매출_금액 + 연령대_40_매출_금액 + 시간대_14_17_매출_금액 + 화요일_매출_금액 + 시간대_17_21_매출_금액'
            }
        }

        # 선택된 상권에 대한 변수 함수 및 모델 생성 ( 대분류 상권 : 골목상권, 발달상권, 전통시장, 관광특구 )
        pred_variables = area_models[BigTradingArea]['variables_func'](
            tradingArea, smallBusiType)
        
        linear_model = smf.ols(formula=area_models[BigTradingArea]['formula'],
                               data=area_data).fit()

        # 예측 결과 계산
        results = {}
        for quarter in ['1분기', '2분기', '3분기', '4분기']:

            # 분기별 매출금액의 예측값
            if quarter in pred_variables.index:
                pred = linear_model.predict(pred_variables.loc[quarter])
                if not pred.dropna().empty:
                    results[quarter] = round(
                        (pred / pred_variables.loc[quarter, '점포수']).values.tolist()[0])
                else:
                    results[quarter] = 0
            else:
                results[quarter] = 0

        # 분석 Report Data
        report_data = report(BigTradingArea, tradingArea, smallBusiType)

        # map.html의 분류 카테고리에 들어갈 Dict
        preData = {
            'businessType': businessType,
            'tradingArea': tradingArea,
            'BigTradingArea': BigTradingArea,
            'smallBusiType': smallBusiType,
            **{f'result{idx}': results[f'{idx}분기'] for idx in range(1, 5)}
        }

        # map.html의 점포수, 성별 매출 비율, 시간대별 매출 비율에 들어갈 Dict
        reportData = {
            'jumpo19': report_data[0]['2019'].values.tolist(),
            'jumpo20': report_data[0]['2020'].values.tolist(),
            'jumpo21': report_data[0]['2021'].values.tolist(),

            'gender19': report_data[1]['2019'].values.tolist(),
            'gender20': report_data[1]['2020'].values.tolist(),
            'gender21': report_data[1]['2021'].values.tolist(),

            'time19': report_data[2]['2019'].values.tolist(),
            'time20': report_data[2]['2020'].values.tolist(),
            'time21': report_data[2]['2021'].values.tolist()
        }

    return JsonResponse({'preData': preData, 'reportData': reportData})

def report(BigTradingArea, tradingArea, smallBusiType):

    # 19~21년 용산구 매출 데이터 통합본
    global yongsan_data

    # 대분류, 소분류 상권, 업종명 분류
    sang = yongsan_data.loc[(yongsan_data['상권_구분_코드_명'] == BigTradingArea) & (yongsan_data['상권_코드_명'] == tradingArea)]
    service = sang[sang['서비스_업종_코드_명'] == smallBusiType]

    # 연도별로 데이터를 분리
    s19, s20, s21 = [service[service['기준_년_코드'] == year]
                     if year in service['기준_년_코드'].values else pd.DataFrame() for year in [2019, 2020, 2021]]

    jumpo19 = [s19[s19['기준_분기_코드'] == i]['점포수'].values[0]
             if i in s19['기준_분기_코드'].values else 0 for i in range(1, 5)]
    jumpo20 = [s20[s20['기준_분기_코드'] == i]['점포수'].values[0]
             if i in s20['기준_분기_코드'].values else 0 for i in range(1, 5)]
    jumpo21 = [s21[s21['기준_분기_코드'] == i]['점포수'].values[0]
             if i in s21['기준_분기_코드'].values else 0 for i in range(1, 5)]
    
    # DataFrame에 입력
    jumpo = pd.DataFrame(zip(jumpo19, jumpo20, jumpo21), 
                       columns=['2019', '2020', '2021'],
                       index=['1분기', '2분기', '3분기', '4분기'])

    # 해당년도의 남.녀 매출 비율
    gender_var = ['남성_매출_비율', '여성_매출_비율']

    if len(s19) != 0:
        gender19 = s19[gender_var].iloc[0].values
    else:
        gender19 = np.array([0, 0])
    if len(s20) != 0:
        gender20 = s20[gender_var].iloc[0].values
    else:
        gender20 = np.array([0, 0])
    if len(s21) != 0:
        gender21 = s21[gender_var].iloc[0].values
    else:
        gender21 = np.array([0, 0])

    # DataFrame에 입력
    gender = pd.DataFrame(zip(gender19, gender20, gender21),
                       columns=['2019', '2020', '2021'],
                       index=['남', '여'])

    # 연도별 시간대별 매출 비율
    time_var = ['시간대_00_06_매출_비율', '시간대_06_11_매출_비율', '시간대_11_14_매출_비율',
                '시간대_14_17_매출_비율', '시간대_17_21_매출_비율', '시간대_21_24_매출_비율']
    empty_var = [0, 0, 0, 0, 0, 0]

    time19 = s19[time_var].iloc[0].values if len(
        s19) != 0 else np.array(empty_var)
    time20 = s20[time_var].iloc[0].values if len(
        s20) != 0 else np.array(empty_var)
    time21 = s21[time_var].iloc[0].values if len(
        s21) != 0 else np.array(empty_var)
    
    # DataFrame에 입력
    time = pd.DataFrame(zip(time19, time20, time21),
                        columns=['2019', '2020', '2021'],
                        index=['00_06', '06_11', '11_14', '14_17', '17_21', '21_24'])

    # 점포수, 
    return jumpo, gender, time

def bal_variables(tradingArea, smallBusiType):

    # 19~21년 용산구 매출 데이터 통합본
    global yongsan_data

    df = yongsan_data[yongsan_data['상권_구분_코드_명'] == "발달상권"]
    sang = df[df['상권_코드_명'] == tradingArea]
    
    # 선택한 업종이 있으면 데이터를 불러오고 없으면 빈 DataFrame을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):

        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명'] == smallBusiType]

        # 모델에 넣어줄 미지의 값이다.(예측값에 사용)
        # 분기별 평균을 구한다.
        avg_data = service.groupby(service['기준_분기_코드']).mean()

        # 예측값에 넣을 변수들만 담기위해 빈 데이터 프레임을 만들어준다.
        predictdata = pd.DataFrame()

        # 산정이 안된 분기가 있을 경우 길이를 모르기때문에 index를 돈다.
        for i in avg_data.index:

            # 0부터 시작하기때문에 -1을 해준다.
            df = pd.DataFrame({'시간대_14_17_매출_금액': avg_data['시간대_14_17_매출_금액'].iloc[i-1],
                              '수요일_매출_금액': avg_data['수요일_매출_금액'].iloc[i-1],
                               '시간대_11_14_매출_금액': avg_data['시간대_11_14_매출_금액'].iloc[i-1],
                               '월요일_매출_금액': avg_data['월요일_매출_금액'].iloc[i-1],
                               '금요일_매출_금액': avg_data['금요일_매출_금액'].iloc[i-1],
                               '점포수': avg_data['점포수'].iloc[i-1]}, index=[str(i)+'분기'])

            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata = pd.concat([predictdata, df])
            
            i+1
        return predictdata
    else:
        return pd.DataFrame()

def gol_variables(tradingArea, smallBusiType):

    # 19~21년 용산구 매출 데이터 통합본
    global yongsan_data

    df = yongsan_data[yongsan_data['상권_구분_코드_명'] == "골목상권"]
    sang = df[df['상권_코드_명'] == tradingArea]

    # 선택한 업종이 있으면 데이터를 불러오고 없으면 빈 DataFrame을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):

        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명'] == smallBusiType]

        # 모델에 넣어줄 미지의 값이다.(예측값에 사용)
        # 분기별 평균을 구한다.
        avg_data = service.groupby(service['기준_분기_코드']).mean()

        # 예측값에 넣을 변수들만 담기위해 빈 데이터 프레임을 만들어준다.
        predictdata = pd.DataFrame()

        # 산정이 안된 분기가 있을 경우 길이를 모르기때문에 index를 돈다.
        for i in avg_data.index:

            # 0부터 시작하기때문에 -1을 해준다.
            df = pd.DataFrame({'월요일_매출_금액': avg_data['월요일_매출_금액'].iloc[i-1],
                              '금요일_매출_금액': avg_data['금요일_매출_금액'].iloc[i-1],
                               '남성_매출_금액': avg_data['남성_매출_금액'].iloc[i-1],
                               '수요일_매출_금액': avg_data['수요일_매출_금액'].iloc[i-1],
                               '화요일_매출_금액': avg_data['화요일_매출_금액'].iloc[i-1],
                               '점포수': avg_data['점포수'].iloc[i-1]}, index=[str(i)+'분기'])

            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata = pd.concat([predictdata, df])
            
            i+1
        return predictdata
    else:
        return pd.DataFrame()

def jeon_variables(tradingArea, smallBusiType):

    # 19~21년 용산구 매출 데이터 통합본
    global yongsan_data

    df = yongsan_data[yongsan_data['상권_구분_코드_명'] == "전통시장"]
    sang = df[df['상권_코드_명'] == tradingArea]

    # 선택한 업종이 있으면 데이터를 불러오고 없으면 빈 DataFrame을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):

        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명'] == smallBusiType]

        # 모델에 넣어줄 미지의 값이다.(예측값에 사용)
        # 분기별 평균을 구한다.
        avg_data = service.groupby(service['기준_분기_코드']).mean()

        # 예측값에 넣을 변수들만 담기위해 빈 데이터 프레임을 만들어준다.
        predictdata = pd.DataFrame()

        # 산정이 안된 분기가 있을 경우 길이를 모르기때문에 index를 돈다.
        for i in avg_data.index:

            # 0부터 시작하기때문에 -1을 해준다.
            df = pd.DataFrame({'금요일_매출_금액': avg_data['금요일_매출_금액'].iloc[i-1],
                              '수요일_매출_금액': avg_data['수요일_매출_금액'].iloc[i-1],
                               '여성_매출_금액': avg_data['여성_매출_금액'].iloc[i-1],
                               '화요일_매출_금액': avg_data['화요일_매출_금액'].iloc[i-1],
                               '목요일_매출_금액': avg_data['목요일_매출_금액'].iloc[i-1],
                               '점포수': avg_data['점포수'].iloc[i-1]}, index=[str(i)+'분기'])

            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata = pd.concat([predictdata, df])
            
            i+1

        return predictdata
    else:
        return pd.DataFrame()

def cul_variables(tradingArea, smallBusiType):

    # 19~21년 용산구 매출 데이터 통합본
    global yongsan_data

    df = yongsan_data[yongsan_data['상권_구분_코드_명'] == "관광특구"]
    sang = df[df['상권_코드_명'] == tradingArea]

    # 선택한 업종이 있으면 데이터를 불러오고 없으면 빈 DataFrame을 리턴한다.
    if smallBusiType in list(sang['서비스_업종_코드_명']):

        # 선택한 서비스업종의 행들만 불러온다.
        service = sang[sang['서비스_업종_코드_명'] == smallBusiType]

        # 모델에 넣어줄 미지의 값이다.(예측값에 사용)
        # 분기별 평균을 구한다.
        avg_data = service.groupby(service['기준_분기_코드']).mean()

        # 예측값에 넣을 변수들만 담기위해 빈 데이터 프레임을 만들어준다.
        predictdata = pd.DataFrame()

        # 산정이 안된 분기가 있을 경우 길이를 모르기때문에 index를 돈다.
        for i in avg_data.index:

            # 0부터 시작하기때문에 -1을 해준다.
            df = pd.DataFrame({'남성_매출_금액': avg_data['남성_매출_금액'].iloc[i-1],
                              '연령대_40_매출_금액': avg_data['연령대_40_매출_금액'].iloc[i-1],
                               '시간대_14_17_매출_금액': avg_data['시간대_14_17_매출_금액'].iloc[i-1],
                               '화요일_매출_금액': avg_data['화요일_매출_금액'].iloc[i-1],
                               '시간대_17_21_매출_금액': avg_data['시간대_17_21_매출_금액'].iloc[i-1],
                               '점포수': avg_data['점포수'].iloc[i-1]}, index=[str(i)+'분기'])

            # 위 행들을 준비해둔 데이터 프레임에 담아준다.
            predictdata = pd.concat([predictdata, df])
            
            i+1
        return predictdata
    else:
        return pd.DataFrame()
