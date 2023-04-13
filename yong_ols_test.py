# -*- coding: utf-8 -*-

import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from statsmodels.stats.outliers_influence import variance_inflation_factor

rc('font', family='malgun gothic')  # 한글 깨짐 방지
warnings.filterwarnings(action='ignore')  # 경고출력안하기

# # Colab 한글 깨짐 현상 방지
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf

pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 데이터 불러오기
df2019 = pd.read_csv(
    'data/서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2019년.csv', encoding='euc-kr')
df2020 = pd.read_csv(
    'data/서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2020년.csv', encoding='euc-kr')
df2021 = pd.read_csv(
    'data/서울시_우리마을가게_상권분석서비스(신_상권_추정매출)_2021년.csv', encoding='euc-kr')

# dfdata = pd.read_csv(
#     "https://raw.githubusercontent.com/Kshinhye/aniorimiroDATA/master/yongsan2021.csv", encoding='utf-8')
# dfdata.to_csv('dfdata.csv')

# print(df2019.head(3), df2019.shape) # (147972, 80)
# print(df2020.head(3), df2020.shape) # (144749, 80)
# print(df2021.head(3), df2021.shape) # (140830, 80)

yongdf = pd.concat(objs=[df2019, df2020, df2021], axis=0)  # 아래로 밀어넣기
columns = yongdf.columns  # 칼럼은 같으니 뽑아서 저장해서 나중에 쓰자
# print(yongdf.columns)
# print(yongdf.info())
# print(yongdf.describe()) 
# 분기매출의 이상치 확인 => 1만원 이하는 제거
yongdf = yongdf.drop(index=[7522, 3205], axis=0)

# df.to_csv('data/2019~2021 서울시.csv') # 데이터 저장

# 각 상권별로 데이터를 분리
# -------------------------------------  용산 골목상권 분류 (42) ---------------------------------------
yong_gol_list = ['NH농협은행 보광동지점', '경리단길남측', '경리단길북측', '남영역 1번', '남정초등학교', '리움미술관', '배문고등학교',
                '삼각지역 14번', '삼각지역 3번', '삼광초등학교', '새남터성지', '서빙고동주민센터', '서빙고역 1번', '서울독일학교',
                '서울역 12번', '서울역 15번', '성심여자고등학교', '숙대입구', '숙대입구역 1번', '열정도', '오산고등학교', '오산중학교',
                '용산구청', '용산세무서', '우사단길', '유엔빌리지길', '이촌동점보아파트', '이태원엔틱가구거리', '이태원역 북측',
                '한강로동땡땡거리(은행나무길)', '한강미주맨션아파트', '한강진역 3번', '한국폴리텍대학서울정수캠퍼스', '한남초등학교',
                '한남힐사이드아파트', '해방촌 남동측', '해방촌예술마을', '효창공원앞역 2번', '효창공원앞역 5번', '효창공원앞역 6번', '효창동주민센터', '후암동주민센터']

gol = yongdf[(yongdf['상권_구분_코드_명'] == '골목상권') & (
    yongdf['상권_코드_명'].isin(yong_gol_list))]
gol.to_csv('data/용산구 골목상권.csv', index=False)  # 저장
print(gol['상권_코드_명'].unique().size)  # 42

# -------------------------------------  용산 발달상권 분류 (7) ---------------------------------------
yong_bal_list = ['남영동 먹자골목', '삼각지역',
                '숙대입구역(남영역, 남영동)', '신용산역(용산역)', '용산전자상가(용산역)', '이태원(이태원역)', '한남오거리']

bal = yongdf[(yongdf['상권_구분_코드_명'] == '발달상권') & (
    yongdf['상권_코드_명'].isin(yong_bal_list))]
bal.to_csv('data/용산구 발달상권.csv', index=False)
print(bal['상권_코드_명'].unique().size)  # 7
# -------------------------------------  용산 전통시장 분류 (7) ---------------------------------------
yong_jeon_list = ['만리시장', '보광시장', '신흥시장', '용산용문시장', '이촌종합시장', '이태원시장', '후암시장']

jeon = yongdf[(yongdf['상권_구분_코드_명'] == '전통시장') & (
    yongdf['상권_코드_명'].isin(yong_jeon_list))]
jeon.to_csv('data/용산구 전통시장.csv', index=False)  # 저장
print(jeon['상권_코드_명'].unique().size)  # 7

# -------------------------------------  용산 관광특구 분류 (3) ---------------------------------------
yong_cul_list = ['이태원 관광특구']
cul = yongdf[(yongdf['상권_구분_코드_명'] == '관광특구') & (
    yongdf['상권_코드_명'].isin(yong_cul_list))]
cul.to_csv('data/용산구 관광특구.csv', index=False)  # 저장
print(cul['상권_코드_명'].unique().size)  # 1

print(gol.shape, gol.isnull().sum())  # (8792, 80) # 결측치 x
print(cul.shape, cul.isnull().sum())  # (550, 80) # 결측치 x
print(jeon.shape, jeon.isnull().sum())  # (1569, 80) # 결측치 x
print(bal.shape, bal.isnull().sum())  # (3135, 80) # 결측치 x
    
# 각 상권의 업종별 매출액을 분기별로 시각화
def calculate_mean(df, year, quarter):

    temp_df = df[(df['기준_년_코드'] == year) & (df['기준_분기_코드'] == quarter)]
    mean_value = temp_df['분기당_매출_금액'].mean()
    temp_df['분기당_매출_금액'] = mean_value
    return temp_df

def plot_service(df, title):
    category = df[["기준_년_코드", "기준_분기_코드", "서비스_업종_코드_명", "분기당_매출_금액"]]
    group_service = category.groupby("서비스_업종_코드_명")
    print(f"{title} 변수 레이블 개수 : {len(group_service)}")

    fig, axes = plt.subplots(7, 9, figsize=(30, 20))
    axes = axes.ravel()

    for idx, (axis, (service, yongdf)) in enumerate(zip(axes, group_service)):
        yongdf_x = (yongdf['기준_년_코드'].astype('str') +
                    '_' + yongdf['기준_분기_코드'].astype('str'))
        a_yongdf = yongdf[yongdf['서비스_업종_코드_명'] == service]

        mean_dfs = [calculate_mean(a_yongdf, year, quarter) for year in range(
            2019, 2022) for quarter in range(1, 5)]
        yongdf_y = pd.concat(mean_dfs)

        axis.plot(yongdf_x, yongdf_y['분기당_매출_금액'])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.set_title(f"{title} : {service}")

    plt.tight_layout()
    plt.show()

# plot_service(gol, "골목상권")
# plot_service(cul, "관광특구")
# plot_service(jeon, "전통시장")
# plot_service(bal, "발달상권")

"""
# 코로나 이전, 이후의 두 집단으로 나누어 T-test를 실시할 예정이었지만
# 정규성을 만족하지 못하므로 데이터를 willcoxon 검정을 실시해보겠음

# 귀무 : COVID-19 초반 확산세가 상권 매출액에 영향을 주지 않는다.
# 대립 : COVID-19 초반 확산세가 상권 매출액에 영향을 준다.

# before_df : 초반 확산세가 오르기전 (19년 1분기 ~ 20년 1분기), after_df : 후 (20년 2분기 ~ 21년 2분기)
# yongdf(용산구 상권 통합데이터)를 기반으로 분리하도록 하겠음

# df_20191 = yongdf[yongdf['기준_년_코드']=='2019-1']
# df_20192 = yongdf[yongdf['기준_년_코드']=='2019-2']
# df_20193 = yongdf[yongdf['기준_년_코드']=='2019-3']
# df_20194 = yongdf[yongdf['기준_년_코드']=='2019-4']
# df_20201 = yongdf[yongdf['기준_년_코드']=='2020-1']
# before_df = pd.concat([df_20191,df_20192,df_20193,df_20194,df_20201],axis=0)
#
# df_20202 = yongdf[yongdf['기준_년_코드']=='2020-2']
# df_20203 = yongdf[yongdf['기준_년_코드']=='2020-3']
# df_20204 = yongdf[yongdf['기준_년_코드']=='2020-4']
# df_20211 = yongdf[yongdf['기준_년_코드']=='2021-1']
# df_20212 = yongdf[yongdf['기준_년_코드']=='2021-2']
# after_df = pd.concat([df_20202,df_20203,df_20204,df_20211,df_20212],axis=0)
#
# print(before_df.head(10),' ',before_df.info()) # 약 18만개의 row
# print(after_df.head(10),' ',after_df.info()) # 약 18만개의 row 합성까지 완료
# print(before_df.describe())
# '''
#            분기당_매출_금액
# count      184335.00000
# mean      598731078.86561
# std      5776190403.96165
# min             97.00000
# 25%     25549884.50000
# 50%       97491942.00000
# 75%    362824363.00000
# max    1157766332654.00000
# '''
# print(after_df.describe())
# '''
#               분기당_매출_금액
# count        179263.00000
# mean       650397640.16530
# std        6588838615.85617
# min                44.00000
# 25%         24678396.00000
# 50%         96000000.00000
# 75%         369538900.50000
# max        1065044715417.00000
# '''

# 이제 이 데이터들로 가설검정을 실시함
# 반복문을 통해 여러번 pvalue값을 확인 (확인해보니 0에 가까운 값들이 나온다)
# for i in range(5):
#     before=[]
#     after=[]
#     print(before) # 리스트가 초기화 되는지 확인
#     for n in range(100):
#         bef_df = before_df['분기당_매출_금액'].sample(n=50000).mean() # 총 데이터의 30% 정도의 양인
#         aft_df = after_df['분기당_매출_금액'].sample(n=50000).mean() # 50000개의 샘플을 뽑아 평균을 만들고
#         before.append(bef_df) # 리스트에 담아서 (100개)
#         after.append(aft_df) # 리스트에 담아서 (100개)
#     print(stats.wilcoxon(before, after).pvalue) # wilcoxon 검정을 통해 pvalue값을 확인 (5번 확인)

# 위의 검정을 통해 pvalue < 0.05 이므로 대립가설 채택
# 대립 : COVID-19 초반 확산세가 상권 매출액에 영향을 준다.

# 본래는 확산세가 증가하면서 매출액의 영향을 끼친 시점의 전후 데이터로 나누어 사용하려고 했지만 데이터의 양이 너무 적어짐
# 이에 따라 COVID-19 확진자가 존재할 때 존재하지 않을때를 독립변수로 활용하기로 했다.
# 확진자의 수가 적을 때와 많을 때의 차이가 심하기 때문에 유의미하지 않다고 생각되어 거리두기를 기준으로 정하기로 했다
# 거리두기로 인해 식당 인원 제한과 마감시간이 제한되었을 때를 1 아닐때를 0인 범주형 변수로 만들기로 했다

# 거리두기 1,2단계를 = 0    /    3,4단계를 = 1이라고 정하기로 했다
# 기준은 사적모임금지 + 5인이상 집합금지 + 22시 이후 이용금지가 함께 적용 되었을때를 3(2.5단계 포함),4단계라고 정하기로함
# 개인 활동 방역수칙 (참조 : https://namu.wiki/w/%EC%82%AC%ED%9A%8C%EC%A0%81%20%EA%B1%B0%EB%A6%AC%EB%91%90%EA%B8%B0/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD)
# 19년 1~ 20년2분기 = 0, 20년 3분기 ~ 21년 1분기 = 1, 21년 2분기 = 0, 21년 3분기 = 1, 21년 4분기 = 0 으로 정했다.
# 기준은 ('covid19/질병관리청_사회적 거리두기 시행연혁-20220901.hwpx') 참조했음
"""

# 상권별로 거리두기 변수를 추가
def covid_variables_add():

    sangs = [gol, bal, jeon, cul]

    for sang in sangs:
        sang['거리두기_단계'] = 0
        for year in range(2019, 2022):
            for quarter in range(1, 5):
                if year == 2020 and (quarter == 3 or quarter == 4):
                    sang.loc[(sang['기준_년_코드'] == year) & (sang['기준_분기_코드'] == quarter), '거리두기_단계'] = 1
                elif year == 2021 and (quarter == 1 or quarter == 3):
                    sang.loc[(sang['기준_년_코드'] == year) & (sang['기준_분기_코드'] == quarter), '거리두기_단계'] = 1


    # gol.to_csv('data/용산구 골목상권(거리두기 추가).csv', index=False)    # 저장해서 확인한 결과 문제없이 들어갔음
    # bal.to_csv('data/용산구 발달상권(거리두기 추가).csv', index=False)
    # jeon.to_csv('data/용산구 전통시장(거리두기 추가).csv', index=False)
    # cul.to_csv('data/용산구 관광특구(거리두기 추가).csv', index=False)

covid_variables_add()

# 시각화
def plot_heatmaps(dataframes, col_ranges, title_suffix, cmap="YlGnBu"):
    f, axes = plt.subplots(1, 4, sharey=True, constrained_layout=True)
    titles = ['골목상권', '발달상권', '전통시장', '관광특구']

    for i, (data, col_range) in enumerate(zip(dataframes, col_ranges)):
        h = sns.heatmap(round(data.iloc[:, col_range[0]:col_range[1]].corrwith(data['분기당_매출_금액']).to_frame(name='분기당_매출_금액'), 2),
                        cmap=cmap, cbar=False, ax=axes[i], annot=True, linewidths=0.3, center=0.5)
        h.set_title(titles[i] + title_suffix)

    plt.show()

def run_plot_heatmaps():

    dataframes = [gol, bal, jeon, cul]
    col_ranges = [(0, 10), (10, 33), (33, 56), (56, 81)]
    title_suffixes = [' 상권', ' 비율', ' 금액', ' 건수']

    for col_range in col_ranges:
        plot_heatmaps(dataframes, [col_range] * 4, title_suffixes[col_ranges.index(col_range)])

# run_plot_heatmaps()

# 모델 (OLS)
# 상권 별로 상관관계가 모두 다르기에 상권 별로 변수를 다르게 사용할 예정
# def multiple_regression_analysis(gol):
#     # 이상치 제거
#     q1 = gol['분기당_매출_금액'].quantile(0.25)
#     q3 = gol['분기당_매출_금액'].quantile(0.75)
#     iqr = q3 - q1
#     condition = gol['분기당_매출_금액'] > q3 + 1.5 * iqr
#     gol.drop(gol[condition].index, inplace=True)

#     # 사용할 변수 설정
#     x = gol[['월요일_매출_금액', '금요일_매출_금액', '남성_매출_금액', '수요일_매출_금액', '화요일_매출_금액']]
#     y = gol['분기당_매출_금액']

#     # 모델 학습
#     lm = smf.ols(formula='y ~ x', data=gol).fit()
#     print(lm.summary())

#     # 잔차확인 (residual)
#     fitted = lm.predict(x)
#     residual = y - fitted
#     print('잔차의 평균:', np.mean(residual))

#     # 정규성 확인
#     sr = stats.zscore(residual)
#     print('shapito test: ', stats.shapiro(residual))
#     (x_score, y_score), _ = stats.probplot(sr)
#     sns.scatterplot(x_score, y_score)
#     plt.plot([-3, 3], [-3, 3], '--')
#     plt.show()

#     # 선형성 확인
#     sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'yellow'})
#     plt.plot([fitted.min(), fitted.max()], [0, 0], '--')
#     plt.show()

#     # 등분산성 확인
#     sns.regplot(fitted, np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
#     plt.show()

#     # 다중공선성 확인
#     vif_df = pd.DataFrame()
#     vif_df['feature'] = x.columns
#     vif_df['vif_value'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
#     print(vif_df)

#     # 독립성 확인 (Durbin-Watson)
#     print('Durbin-Watson:', statsmodels.stats.stattools.durbin_watson(lm.resid))

#     return lm

# def multiple_regression_analysis(bal):
#     # 이상치 제거
#     q1 = bal['분기당_매출_금액'].quantile(0.25)
#     q3 = bal['분기당_매출_금액'].quantile(0.75)
#     iqr = q3 - q1
#     condition = bal['분기당_매출_금액'] > q3 + 1.5 * iqr
#     bal.drop(bal[condition].index, inplace=True)

#     # 사용할 변수 설정
#     x = bal[['시간대_14_17_매출_금액','수요일_매출_금액','시간대_11_14_매출_금액','월요일_매출_금액','금요일_매출_금액']]
#     y = bal['분기당_매출_금액']

#     # 모델 학습
#     lm = smf.ols(formula='y ~ x', data=bal).fit()
#     print(lm.summary())

#     # 잔차확인 (residual)
#     fitted = lm.predict(x)
#     residual = y - fitted
#     print('잔차의 평균:', np.mean(residual))

#     # 정규성 확인
#     sr = stats.zscore(residual)
#     print('shapito test: ', stats.shapiro(residual))
#     (x_score, y_score), _ = stats.probplot(sr)
#     sns.scatterplot(x_score, y_score)
#     plt.plot([-3, 3], [-3, 3], '--')
#     plt.show()

#     # 선형성 확인
#     sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'yellow'})
#     plt.plot([fitted.min(), fitted.max()], [0, 0], '--')
#     plt.show()

#     # 등분산성 확인
#     sns.regplot(fitted, np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
#     plt.show()

#     # 다중공선성 확인
#     vif_df = pd.DataFrame()
#     vif_df['feature'] = x.columns
#     vif_df['vif_value'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
#     print(vif_df)

#     # 독립성 확인 (Durbin-Watson)
#     print('Durbin-Watson:', statsmodels.stats.stattools.durbin_watson(lm.resid))

#     return lm

# def multiple_regression_analysis(jeon):
#     # 이상치 제거
#     q1 = jeon['분기당_매출_금액'].quantile(0.25)
#     q3 = jeon['분기당_매출_금액'].quantile(0.75)
#     iqr = q3 - q1
#     condition = jeon['분기당_매출_금액'] > q3 + 1.5 * iqr
#     jeon.drop(jeon[condition].index, inplace=True)

#     # 사용할 변수 설정
#     x = jeon[['금요일_매출_금액','수요일_매출_금액','여성_매출_금액','화요일_매출_금액','목요일_매출_금액']]
#     y = jeon['분기당_매출_금액']

#     # 모델 학습
#     lm = smf.ols(formula='y ~ x', data=jeon).fit()
#     print(lm.summary())

#     # 잔차확인 (residual)
#     fitted = lm.predict(x)
#     residual = y - fitted
#     print('잔차의 평균:', np.mean(residual))

#     # 정규성 확인
#     sr = stats.zscore(residual)
#     print('shapito test: ', stats.shapiro(residual))
#     (x_score, y_score), _ = stats.probplot(sr)
#     sns.scatterplot(x_score, y_score)
#     plt.plot([-3, 3], [-3, 3], '--')
#     plt.show()

#     # 선형성 확인
#     sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'yellow'})
#     plt.plot([fitted.min(), fitted.max()], [0, 0], '--')
#     plt.show()

#     # 등분산성 확인
#     sns.regplot(fitted, np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
#     plt.show()

#     # 다중공선성 확인
#     vif_df = pd.DataFrame()
#     vif_df['feature'] = x.columns
#     vif_df['vif_value'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
#     print(vif_df)

#     # 독립성 확인 (Durbin-Watson)
#     print('Durbin-Watson:', statsmodels.stats.stattools.durbin_watson(lm.resid))

#     return lm

# def multiple_regression_analysis(cul):
    # # 이상치 제거
    # q1 = cul['분기당_매출_금액'].quantile(0.25)
    # q3 = cul['분기당_매출_금액'].quantile(0.75)
    # iqr = q3 - q1
    # condition = cul['분기당_매출_금액'] > q3 + 1.5 * iqr
    # cul.drop(cul[condition].index, inplace=True)

    # # 사용할 변수 설정
    # x = cul[['남성_매출_금액','연령대_40_매출_금액','시간대_14_17_매출_금액','화요일_매출_금액','시간대_17_21_매출_금액']]
    # y = cul['분기당_매출_금액']

    # # 모델 학습
    # lm = smf.ols(formula='y ~ x', data=cul).fit()
    # print(lm.summary())

    # # 잔차확인 (residual)
    # fitted = lm.predict(x)
    # residual = y - fitted
    # print('잔차의 평균:', np.mean(residual))

    # # 정규성 확인
    # sr = stats.zscore(residual)
    # print('shapito test: ', stats.shapiro(residual))
    # (x_score, y_score), _ = stats.probplot(sr)
    # sns.scatterplot(x_score, y_score)
    # plt.plot([-3, 3], [-3, 3], '--')
    # plt.show()

    # # 선형성 확인
    # sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'yellow'})
    # plt.plot([fitted.min(), fitted.max()], [0, 0], '--')
    # plt.show()

    # # 등분산성 확인
    # sns.regplot(fitted, np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
    # plt.show()

    # # 다중공선성 확인
    # vif_df = pd.DataFrame()
    # vif_df['feature'] = x.columns
    # vif_df['vif_value'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    # print(vif_df)

    # # 독립성 확인 (Durbin-Watson)
    # print('Durbin-Watson:', statsmodels.stats.stattools.durbin_watson(lm.resid))

    # return lm

def multiple_regression_analysis(df, x_variables):

    # 이상치 제거
    q1 = df['분기당_매출_금액'].quantile(0.25)
    q3 = df['분기당_매출_금액'].quantile(0.75)
    iqr = q3 - q1
    condition = df['분기당_매출_금액'] > q3 + 1.5 * iqr
    df.drop(df[condition].index, inplace=True)

    # 사용할 변수 설정
    x = df[x_variables]
    y = df['분기당_매출_금액']

    # 모델 학습
    lm = smf.ols(formula='y ~ x', data=df).fit()
    print(lm.summary())

    # 잔차확인 (residual)
    fitted = lm.predict(x)
    residual = y - fitted
    print('잔차의 평균:', np.mean(residual))

    # 정규성 확인
    sr = stats.zscore(residual)
    print('shapito test: ', stats.shapiro(residual))
    (x_score, y_score), _ = stats.probplot(sr)
    sns.scatterplot(x_score, y_score)
    plt.plot([-3, 3], [-3, 3], '--')
    plt.show()

    # 선형성 확인
    sns.regplot(fitted, residual, lowess=True, line_kws={'color': 'yellow'})
    plt.plot([fitted.min(), fitted.max()], [0, 0], '--')
    plt.show()

    # 등분산성 확인
    sns.regplot(fitted, np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
    plt.show()

    # 다중공선성 확인
    vif_df = pd.DataFrame()
    vif_df['feature'] = x.columns
    vif_df['vif_value'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    print(vif_df)

    # 독립성 확인 (Durbin-Watson)
    print('Durbin-Watson:', statsmodels.stats.stattools.durbin_watson(lm.resid))

    return lm

def gol_analysis():

    gol_variables = ['월요일_매출_금액', '금요일_매출_금액', '남성_매출_금액', '수요일_매출_금액', '화요일_매출_금액']
    
    gol_model = multiple_regression_analysis(gol, gol_variables)

    return gol_model

def bal_analysis():

    bal_variables = ['시간대_14_17_매출_금액', '수요일_매출_금액', '시간대_11_14_매출_금액', '월요일_매출_금액', '금요일_매출_금액']
    
    bal_model = multiple_regression_analysis(bal, bal_variables)

    return bal_model

def jeon_analysis():

    jeon_variables = ['금요일_매출_금액', '수요일_매출_금액', '여성_매출_금액', '화요일_매출_금액', '목요일_매출_금액']
    
    jeon_model = multiple_regression_analysis(jeon, jeon_variables)

    return jeon_model

def cul_analysis():

    cul_variables = ['남성_매출_금액','연령대_40_매출_금액','시간대_14_17_매출_금액','화요일_매출_금액','시간대_17_21_매출_금액']
    
    cul_model = multiple_regression_analysis(cul, cul_variables)

    return cul_model

gol_analysis()
# bal_analysis()
# jeon_analysis()
# cul_analysis()

# 모델저장
# import pickle
# pickle.dump(lm, open('gol_model.pickle',mode='wb'))

# 요청을 받아와서 상권분석 실행
# def calldbFunc(request):

#     if request.method=="POST":
#         print('view옴')
#         tradingArea=request.POST.get('tradingArea')
#         BigTradingArea=request.POST.get('BigTradingArea')
#         businessType=request.POST.get('businessType')
#         smallBusiType=request.POST.get('smallBusiType')


#         print(BigTradingArea)
#         print(tradingArea)
#         print(smallBusiType)


#         # 요청을 받으면 해당 상권에 맞는 코드 실행하기
#         # 발달상권 예측모델 실행
#         if BigTradingArea == '발달상권':

#             #발달상권
#             bal=pd.read_csv("https://raw.githubusercontent.com/Choizard/aniorimiro/master/aniorimiro/static/data/%EC%9A%A9%EC%82%B0%EA%B5%AC_%EB%B0%9C%EB%8B%AC%EC%83%81%EA%B6%8C_%EC%9D%B8%EC%BD%94%EB%94%A9.csv", encoding='utf-8')
#             print('발달상권 매출 예상')
#             sang_bal = bal[bal['상권_코드_명']==tradingArea]
#             # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출

#             if smallBusiType in list(sang_bal['서비스_업종_코드_명']):
#                 service_bal = sang_bal[sang_bal['서비스_업종_코드_명']==smallBusiType]
#                 # 업종의 분기별 평균을 토대로 예상매출액을 계산
#                 bal_1 = service_bal[service_bal['기준_분기_코드']==1]
#                 bal_1_a = bal_1['월요일_매출_금액'].mean()
#                 bal_1_b = bal_1['토요일_매출_금액'].mean()
#                 bal_1_c = bal_1['일요일_매출_금액'].mean()
#                 bal_1_mon = bal_1['월요일_매출_건수'].mean()
#                 bal_1_sat = bal_1['토요일_매출_건수'].mean()
#                 bal_1_sn = bal_1['일요일_매출_건수'].mean()

#                 bal_2 = service_bal[service_bal['기준_분기_코드']==2]
#                 bal_2_a = bal_2['월요일_매출_금액'].mean()
#                 bal_2_b = bal_2['토요일_매출_금액'].mean()
#                 bal_2_c = bal_2['일요일_매출_금액'].mean()
#                 bal_2_mon = bal_2['월요일_매출_건수'].mean()
#                 bal_2_sat = bal_2['토요일_매출_건수'].mean()
#                 bal_2_sn = bal_2['일요일_매출_건수'].mean()

#                 bal_3 = service_bal[service_bal['기준_분기_코드']==3]
#                 bal_3_a = bal_3['월요일_매출_금액'].mean()
#                 bal_3_b = bal_3['토요일_매출_금액'].mean()
#                 bal_3_c = bal_3['일요일_매출_금액'].mean()
#                 bal_3_mon = bal_3['월요일_매출_건수'].mean()
#                 bal_3_sat = bal_3['토요일_매출_건수'].mean()
#                 bal_3_sn = bal_3['일요일_매출_건수'].mean()

#                 bal_4 = service_bal[service_bal['기준_분기_코드']==4]
#                 bal_4_a = bal_4['월요일_매출_금액'].mean()
#                 bal_4_b = bal_4['토요일_매출_금액'].mean()
#                 bal_4_c = bal_4['일요일_매출_금액'].mean()
#                 bal_4_mon = bal_4['월요일_매출_건수'].mean()
#                 bal_4_sat = bal_4['토요일_매출_건수'].mean()
#                 bal_4_sn = bal_4['일요일_매출_건수'].mean()

#                 # 모델
#                 model_bal = smf.ols(formula = '분기당_매출_금액 ~ 서비스_업종_코드 + 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service_bal).fit()

#                 # 1분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_bal.values[0][7]],
#                                     '월요일_매출_금액':[bal_1_a],'토요일_매출_금액':[bal_1_b],'일요일_매출_금액':[bal_1_c],
#                                     '월요일_매출_건수':[bal_1_mon],'토요일_매출_건수':[bal_1_sat],'일요일_매출_건수':[bal_1_sn]})
#                 pred1 = model_bal.predict(x_new2)
#                 result1 = (pred1 / bal_1['점포수'].mean()).values[0]
#                 result1 = np.nan_to_num(result1)
#                 print('실제값 1:',bal_1['분기당_매출_금액'].mean() / bal_1['점포수'].mean())
#                 print('예측값 1:',pred1 / bal_1['점포수'].mean())


#                 # 2분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_bal.values[0][7]],
#                                     '월요일_매출_금액':[bal_2_a],'토요일_매출_금액':[bal_2_b],'일요일_매출_금액':[bal_2_c],
#                                     '월요일_매출_건수':[bal_2_mon],'토요일_매출_건수':[bal_2_sat],'일요일_매출_건수':[bal_2_sn]})
#                 pred2 = model_bal.predict(x_new2)
#                 result2 = (pred2 / bal_2['점포수'].mean()).values[0]
#                 result2 = np.nan_to_num(result2)
#                 print('실제값 2:',(bal_2['분기당_매출_금액'].mean()) / bal_2['점포수'].mean())
#                 print('예측값 2:',pred2 / bal_2['점포수'].mean())


#                 # 3분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_bal.values[0][7]],
#                                     '월요일_매출_금액':[bal_3_a],'토요일_매출_금액':[bal_3_b],'일요일_매출_금액':[bal_3_c],
#                                     '월요일_매출_건수':[bal_3_mon],'토요일_매출_건수':[bal_3_sat],'일요일_매출_건수':[bal_3_sn]})
#                 pred3 = model_bal.predict(x_new2)
#                 result3 = (pred3 / bal_3['점포수'].mean()).values[0]
#                 result3 = np.nan_to_num(result3)
#                 print('실제값 3:',(bal_3['분기당_매출_금액'].mean()) / bal_3['점포수'].mean())
#                 print('예측값 3:',pred3 / bal_3['점포수'].mean())


#                 # 4분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_bal.values[0][7]],
#                                     '월요일_매출_금액':[bal_4_a],'토요일_매출_금액':[bal_4_b],'일요일_매출_금액':[bal_4_c],
#                                     '월요일_매출_건수':[bal_4_mon],'토요일_매출_건수':[bal_4_sat],'일요일_매출_건수':[bal_4_sn]})
#                 pred4 = model_bal.predict(x_new2)
#                 result4 = (pred4 / bal_4['점포수'].mean()).values[0]
#                 result4 = np.nan_to_num(result4)
#                 print('실제값 4:',(bal_4['분기당_매출_금액'].mean()) / bal_4['점포수'].mean())
#                 print('예측값 4:',pred4 / bal_4['점포수'].mean())

#                 print(model_bal.summary())
#             else :
#                 result1 = 0
#                 result2 = 0
#                 result3 = 0
#                 result4 = 0
#         # 관광특구 예측모델 실행
#         elif BigTradingArea == '관광특구':

#             #관광특구
#             cul=pd.read_csv("https://raw.githubusercontent.com/Choizard/aniorimiro/master/aniorimiro/static/data/%EC%9A%A9%EC%82%B0%EA%B5%AC_%EA%B4%80%EA%B4%91%ED%8A%B9%EA%B5%AC_%EC%9D%B8%EC%BD%94%EB%94%A9.csv", encoding='utf-8')
#             print('관광특구 매출 예상')
#             sang_cul = cul[cul['상권_코드_명']==tradingArea]
#             # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출

#             if smallBusiType in list(sang_cul['서비스_업종_코드_명']):
#                 service_cul = sang_cul[sang_cul['서비스_업종_코드_명']==smallBusiType]

#                 # 업종의 분기별 평균을 토대로 예상매출액을 계산
#                 cul_1 = service_cul[service_cul['기준_분기_코드']==1]
#                 cul_1_a = cul_1['월요일_매출_금액'].mean()
#                 cul_1_b = cul_1['토요일_매출_금액'].mean()
#                 cul_1_c = cul_1['일요일_매출_금액'].mean()
#                 cul_1_mon = cul_1['월요일_매출_건수'].mean()
#                 cul_1_sat = cul_1['토요일_매출_건수'].mean()
#                 cul_1_sn = cul_1['일요일_매출_건수'].mean()

#                 cul_2 = service_cul[service_cul['기준_분기_코드']==2]
#                 cul_2_a = cul_2['월요일_매출_금액'].mean()
#                 cul_2_b = cul_2['토요일_매출_금액'].mean()
#                 cul_2_c = cul_2['일요일_매출_금액'].mean()
#                 cul_2_mon = cul_2['월요일_매출_건수'].mean()
#                 cul_2_sat = cul_2['토요일_매출_건수'].mean()
#                 cul_2_sn = cul_2['일요일_매출_건수'].mean()

#                 cul_3 = service_cul[service_cul['기준_분기_코드']==3]
#                 cul_3_a = cul_3['월요일_매출_금액'].mean()
#                 cul_3_b = cul_3['토요일_매출_금액'].mean()
#                 cul_3_c = cul_3['일요일_매출_금액'].mean()
#                 cul_3_mon = cul_3['월요일_매출_건수'].mean()
#                 cul_3_sat = cul_3['토요일_매출_건수'].mean()
#                 cul_3_sn = cul_3['일요일_매출_건수'].mean()

#                 cul_4 = service_cul[service_cul['기준_분기_코드']==4]
#                 cul_4_a = cul_4['월요일_매출_금액'].mean()
#                 cul_4_b = cul_4['토요일_매출_금액'].mean()
#                 cul_4_c = cul_4['일요일_매출_금액'].mean()
#                 cul_4_mon = cul_4['월요일_매출_건수'].mean()
#                 cul_4_sat = cul_4['토요일_매출_건수'].mean()
#                 cul_4_sn = cul_4['일요일_매출_건수'].mean()

#                 # 모델
#                 model_cul = smf.ols(formula = '분기당_매출_금액 ~ 서비스_업종_코드 + 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service_cul).fit()

#                 # 1분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_cul.values[0][7]],
#                                     '월요일_매출_금액':[cul_1_a],'토요일_매출_금액':[cul_1_b],'일요일_매출_금액':[cul_1_c],
#                                     '월요일_매출_건수':[cul_1_mon],'토요일_매출_건수':[cul_1_sat],'일요일_매출_건수':[cul_1_sn]})
#                 pred1 = model_cul.predict(x_new2)
#                 result1 = (pred1 / cul_1['점포수'].mean()).values[0]
#                 result1 = np.nan_to_num(result1)
#                 print('실제값 1:',(cul_1['분기당_매출_금액'].mean()) / cul_1['점포수'].mean())
#                 print('예측값 1:',pred1 / cul_1['점포수'].mean())


#                 # 2분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_cul.values[0][7]],
#                                     '월요일_매출_금액':[cul_2_a],'토요일_매출_금액':[cul_2_b],'일요일_매출_금액':[cul_2_c],
#                                     '월요일_매출_건수':[cul_2_mon],'토요일_매출_건수':[cul_2_sat],'일요일_매출_건수':[cul_2_sn]})
#                 pred2 = model_cul.predict(x_new2)
#                 result2 = (pred2 / cul_2['점포수'].mean()).values[0]
#                 result2 = np.nan_to_num(result2)
#                 print('실제값 2:',(cul_2['분기당_매출_금액'].mean()) / cul_2['점포수'].mean())
#                 print('예측값 2:',pred2 / cul_2['점포수'].mean())


#                 # 3분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_cul.values[0][7]],
#                                     '월요일_매출_금액':[cul_3_a],'토요일_매출_금액':[cul_3_b],'일요일_매출_금액':[cul_3_c],
#                                     '월요일_매출_건수':[cul_3_mon],'토요일_매출_건수':[cul_3_sat],'일요일_매출_건수':[cul_3_sn]})
#                 pred3 = model_cul.predict(x_new2)
#                 result3 = (pred3 / cul_3['점포수'].mean()).values[0]
#                 result3 = np.nan_to_num(result3)
#                 print('실제값 3:',(cul_3['분기당_매출_금액'].mean()) / cul_3['점포수'].mean())
#                 print('예측값 3:',pred3 / cul_3['점포수'].mean())


#                 # 4분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_cul.values[0][7]],
#                                     '월요일_매출_금액':[cul_4_a],'토요일_매출_금액':[cul_4_b],'일요일_매출_금액':[cul_4_c],
#                                     '월요일_매출_건수':[cul_4_mon],'토요일_매출_건수':[cul_4_sat],'일요일_매출_건수':[cul_4_sn]})
#                 pred4 = model_cul.predict(x_new2)
#                 result4 = (pred4 / cul_4['점포수'].mean()).values[0]
#                 result4 = np.nan_to_num(result4)
#                 print('실제값 4:',(cul_4['분기당_매출_금액'].mean()) / cul_4['점포수'].mean())
#                 print('예측값 4:',pred4 / cul_4['점포수'].mean())

#                 print(model_cul.summary())
#             else :
#                 result1 = 0
#                 result2 = 0
#                 result3 = 0
#                 result4 = 0
#         # 골목상권 예측모델 실행
#         elif BigTradingArea == '골목상권':

#             #골목상권
#             gol=pd.read_csv("https://raw.githubusercontent.com/Choizard/aniorimiro/master/aniorimiro/static/data/%EC%9A%A9%EC%82%B0%EA%B5%AC_%EA%B3%A8%EB%AA%A9%EC%83%81%EA%B6%8C_%EC%9D%B8%EC%BD%94%EB%94%A9.csv", encoding='utf-8')
#             print('골목상권 매출 예상')
#             sang_gol = gol[gol['상권_코드_명']==tradingArea]
#             # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출

#             if smallBusiType in list(sang_gol['서비스_업종_코드_명']):
#                 service_gol = sang_gol[sang_gol['서비스_업종_코드_명']==smallBusiType]

#                 # 업종의 분기별 평균을 토대로 예상매출액을 계산
#                 gol_1 = service_gol[service_gol['기준_분기_코드']==1]
#                 gol_1_a = gol_1['월요일_매출_금액'].mean()
#                 gol_1_b = gol_1['토요일_매출_금액'].mean()
#                 gol_1_c = gol_1['일요일_매출_금액'].mean()
#                 gol_1_mon = gol_1['월요일_매출_건수'].mean()
#                 gol_1_sat = gol_1['토요일_매출_건수'].mean()
#                 gol_1_sn = gol_1['일요일_매출_건수'].mean()

#                 gol_2 = service_gol[service_gol['기준_분기_코드']==2]
#                 gol_2_a = gol_2['월요일_매출_금액'].mean()
#                 gol_2_b = gol_2['토요일_매출_금액'].mean()
#                 gol_2_c = gol_2['일요일_매출_금액'].mean()
#                 gol_2_mon = gol_2['월요일_매출_건수'].mean()
#                 gol_2_sat = gol_2['토요일_매출_건수'].mean()
#                 gol_2_sn = gol_2['일요일_매출_건수'].mean()

#                 gol_3 = service_gol[service_gol['기준_분기_코드']==3]
#                 gol_3_a = gol_3['월요일_매출_금액'].mean()
#                 gol_3_b = gol_3['토요일_매출_금액'].mean()
#                 gol_3_c = gol_3['일요일_매출_금액'].mean()
#                 gol_3_mon = gol_3['월요일_매출_건수'].mean()
#                 gol_3_sat = gol_3['토요일_매출_건수'].mean()
#                 gol_3_sn = gol_3['일요일_매출_건수'].mean()

#                 gol_4 = service_gol[service_gol['기준_분기_코드']==4]
#                 gol_4_a = gol_4['월요일_매출_금액'].mean()
#                 gol_4_b = gol_4['토요일_매출_금액'].mean()
#                 gol_4_c = gol_4['일요일_매출_금액'].mean()
#                 gol_4_mon = gol_4['월요일_매출_건수'].mean()
#                 gol_4_sat = gol_4['토요일_매출_건수'].mean()
#                 gol_4_sn = gol_4['일요일_매출_건수'].mean()
#                 # 모델
#                 model_gol = smf.ols(formula = '분기당_매출_금액 ~ 서비스_업종_코드 + 월요일_매출_금액 + 토요일_매출_금액 + 일요일_매출_금액 + 월요일_매출_건수 + 토요일_매출_건수 + 일요일_매출_건수',data = service_gol).fit()

#                 # 1분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_gol.values[0][7]],
#                                     '월요일_매출_금액':[gol_1_a],'토요일_매출_금액':[gol_1_b],'일요일_매출_금액':[gol_1_c],
#                                     '월요일_매출_건수':[gol_1_mon],'토요일_매출_건수':[gol_1_sat],'일요일_매출_건수':[gol_1_sn]})
#                 pred1 = model_gol.predict(x_new2)
#                 result1 = (pred1 / gol_1['점포수'].mean()).values[0]
#                 result1 = np.nan_to_num(result1)
#                 print('실제값 1:',(gol_1['분기당_매출_금액'].mean()) / gol_1['점포수'].mean())
#                 print('예측값 1:',pred1 / gol_1['점포수'].mean())


#                 # 2분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_gol.values[0][7]],
#                                     '월요일_매출_금액':[gol_2_a],'토요일_매출_금액':[gol_2_b],'일요일_매출_금액':[gol_2_c],
#                                     '월요일_매출_건수':[gol_2_mon],'토요일_매출_건수':[gol_2_sat],'일요일_매출_건수':[gol_2_sn]})
#                 pred2 = model_gol.predict(x_new2)
#                 result2 = (pred2 / gol_2['점포수'].mean()).values[0]
#                 result2 = np.nan_to_num(result2)
#                 print('실제값 2:',(gol_2['분기당_매출_금액'].mean()) / gol_2['점포수'].mean())
#                 print('예측값 2:',pred2 / gol_2['점포수'].mean())


#                 # 3분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_gol.values[0][7]],
#                                     '월요일_매출_금액':[gol_3_a],'토요일_매출_금액':[gol_3_b],'일요일_매출_금액':[gol_3_c],
#                                     '월요일_매출_건수':[gol_3_mon],'토요일_매출_건수':[gol_3_sat],'일요일_매출_건수':[gol_3_sn]})
#                 pred3 = model_gol.predict(x_new2)
#                 result3 = (pred3 / gol_3['점포수'].mean()).values[0]
#                 result3 = np.nan_to_num(result3)
#                 print('실제값 3:',(gol_3['분기당_매출_금액'].mean()) / gol_3['점포수'].mean())
#                 print('예측값 3:',pred3 / gol_3['점포수'].mean())


#                 # 4분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_gol.values[0][7]],
#                                     '월요일_매출_금액':[gol_4_a],'토요일_매출_금액':[gol_4_b],'일요일_매출_금액':[gol_4_c],
#                                     '월요일_매출_건수':[gol_4_mon],'토요일_매출_건수':[gol_4_sat],'일요일_매출_건수':[gol_4_sn]})
#                 pred4 = model_gol.predict(x_new2)
#                 result4 = (pred4 / gol_4['점포수'].mean()).values[0]
#                 result4 = np.nan_to_num(result4)
#                 print('실제값 4:',(gol_4['분기당_매출_금액'].mean()) / gol_4['점포수'].mean())
#                 print('예측값 4:',pred4 / gol_4['점포수'].mean())

#                 print(model_gol.summary())
#             else :
#                 result1 = 0
#                 result2 = 0
#                 result3 = 0
#                 result4 = 0
#         # 전통시장 예측모델 실행
#         elif BigTradingArea == '전통시장':

#             #전통시장
#             jeon=pd.read_csv("https://raw.githubusercontent.com/Choizard/aniorimiro/master/aniorimiro/static/data/%EC%9A%A9%EC%82%B0%EA%B5%AC_%EC%A0%84%ED%86%B5%EC%8B%9C%EC%9E%A5_%EC%9D%B8%EC%BD%94%EB%94%A9.csv", encoding='utf-8')
#             print('전통시장 매출 예상')
#             sang_jeon = jeon[jeon['상권_코드_명']==tradingArea]
#             # 존재하지 않는 업종이 선택되어 데이터가 없다면 '데이터가 없습니다' 로 도출

#             if smallBusiType in list(sang_jeon['서비스_업종_코드_명']):
#                 service_jeon = sang_jeon[sang_jeon['서비스_업종_코드_명']==smallBusiType]

#                 # 업종의 분기별 평균을 토대로 예상매출액을 계산
#                 jeon_1 = service_jeon[service_jeon['기준_분기_코드']==1]
#                 jeon_1_a = jeon_1['월요일_매출_금액'].mean()
#                 jeon_1_b = jeon_1['수요일_매출_금액'].mean()
#                 jeon_1_c = jeon_1['목요일_매출_금액'].mean()
#                 jeon_1_bun = jeon_1['분기당_매출_건수'].mean()

#                 jeon_2 = service_jeon[service_jeon['기준_분기_코드']==2]
#                 jeon_2_a = jeon_2['월요일_매출_금액'].mean()
#                 jeon_2_b = jeon_2['수요일_매출_금액'].mean()
#                 jeon_2_c = jeon_2['목요일_매출_금액'].mean()
#                 jeon_2_bun = jeon_2['분기당_매출_건수'].mean()

#                 jeon_3 = service_jeon[service_jeon['기준_분기_코드']==3]
#                 jeon_3_a = jeon_3['월요일_매출_금액'].mean()
#                 jeon_3_b = jeon_3['수요일_매출_금액'].mean()
#                 jeon_3_c = jeon_3['목요일_매출_금액'].mean()
#                 jeon_3_bun = jeon_3['분기당_매출_건수'].mean()

#                 jeon_4 = service_jeon[service_jeon['기준_분기_코드']==4]
#                 jeon_4_a = jeon_4['월요일_매출_금액'].mean()
#                 jeon_4_b = jeon_4['수요일_매출_금액'].mean()
#                 jeon_4_c = jeon_4['목요일_매출_금액'].mean()
#                 jeon_4_bun = jeon_4['분기당_매출_건수'].mean()
#                 # 모델
#                 model_jeon = smf.ols(formula = '분기당_매출_금액 ~ 월요일_매출_금액 + 수요일_매출_금액 + 목요일_매출_금액 + 분기당_매출_건수',data = service_jeon).fit()

#                 # 1분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_jeon.values[0][7]],
#                                     '월요일_매출_금액':[jeon_1_a],'수요일_매출_금액':[jeon_1_b],'목요일_매출_금액':[jeon_1_c],
#                                     '분기당_매출_건수':[jeon_1_bun]})
#                 pred1 = model_jeon.predict(x_new2)
#                 result1 = (pred1 / jeon_1['점포수'].mean()).values[0]
#                 result1 = np.nan_to_num(result1)
#                 print('실제값 1:',(jeon_1['분기당_매출_금액'].mean()) / jeon_1['점포수'].mean())
#                 print('예측값 1:',pred1 / jeon_1['점포수'].mean())


#                 # 2분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_jeon.values[0][7]],
#                                     '월요일_매출_금액':[jeon_2_a],'수요일_매출_금액':[jeon_2_b],'목요일_매출_금액':[jeon_2_c],
#                                     '분기당_매출_건수':[jeon_2_bun]})
#                 pred2 = model_jeon.predict(x_new2)
#                 result2 = (pred2 / jeon_2['점포수'].mean()).values[0]
#                 result2 = np.nan_to_num(result2)
#                 print('실제값 2:',(jeon_2['분기당_매출_금액'].mean()) / jeon_2['점포수'].mean())
#                 print('예측값 2:',pred2 / jeon_2['점포수'].mean())


#                 # 3분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_jeon.values[0][7]],
#                                     '월요일_매출_금액':[jeon_3_a],'수요일_매출_금액':[jeon_3_b],'목요일_매출_금액':[jeon_3_c],
#                                     '분기당_매출_건수':[jeon_3_bun]})
#                 pred3 = model_jeon.predict(x_new2)
#                 result3 = (pred3 / jeon_3['점포수'].mean()).values[0]
#                 result3 = np.nan_to_num(result3)
#                 print('실제값 3:',(jeon_3['분기당_매출_금액'].mean()) / jeon_3['점포수'].mean())
#                 print('예측값 3:',pred3 / jeon_3['점포수'].mean())

#                 # 4분기 매출
#                 x_new2 = pd.DataFrame({'서비스_업종_코드':[service_jeon.values[0][7]],
#                                     '월요일_매출_금액':[jeon_4_a],'수요일_매출_금액':[jeon_4_b],'목요일_매출_금액':[jeon_4_c],
#                                     '분기당_매출_건수':[jeon_4_bun]})
#                 pred4 = model_jeon.predict(x_new2)
#                 result4 = (pred4 / jeon_4['점포수'].mean()).values[0]
#                 result4 = np.nan_to_num(result4)
#                 print('실제값 4:',(jeon_4['분기당_매출_금액'].mean()) / jeon_4['점포수'].mean())
#                 print('예측값 4:',pred4 / jeon_4['점포수'].mean())

#                 print(model_jeon.summary())
#             else :
#                 result1 = 0
#                 result2 = 0
#                 result3 = 0
#                 result4 = 0

#         context = {
#           'businessType':businessType,
#           'tradingArea':tradingArea,
#           'BigTradingArea':BigTradingArea,
#           'smallBusiType':smallBusiType,
#           'result1':result1,
#           'result2':result2,
#           'result3':result3,
#           'result4':result4
#         }

#     # return JsonResponse(context)

