# 코로나 확진자 동향 알아보기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='malgun gothic')  # Matplotlip 한글 깨짐 방지

pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# 데이터 불러오기
y_covid = pd.read_csv('covid19/서울시 코로나19 자치구별 확진자 발생동향.csv',
                      usecols=['자치구 기준일', '용산구 추가', '용산구 전체'], encoding='euc-kr')
s_covid = pd.read_csv('covid19/서울시 코로나19 확진자 발생동향.csv',
                      usecols=['서울시 확진자', '서울시 추가 확진', '전국 확진', '전국 추가 확진'], encoding='euc-kr')
print(y_covid.head(3), y_covid.shape) 
print(s_covid.head(3), s_covid.shape) 

# 옆으로 붙이기
covid = pd.concat([y_covid, s_covid], axis=1)

start_date = pd.to_datetime('2020-02-05')  # 시작 날짜
end_date = pd.to_datetime('2022-12-16')  # 마지막 날짜
dates = pd.date_range(start_date, end_date, freq='D').sort_values(ascending=False)
dates = dates.strftime('%Y%m%d')
dates = pd.DataFrame(dates, columns=['자치구 기준일'])

# 날짜 끼워넣기
covid['자치구 기준일'] = dates['자치구 기준일']
# print(covid.isnull())
covid = covid.dropna()  # 결측치 제거
covid = covid.drop([0])
# print(covid.head(10),' ',covid.info())

# 정렬 오름차순
covid = covid.sort_values(['자치구 기준일'], axis=0)
print(covid.describe())


# 추세 시각화 (log함수 on/off)
def plot_subplot(ax, x, y, title, label, plot_type, color):
    if plot_type == 'bar':
        ax.bar(x, y, label=label, color=color)
    elif plot_type == 'line':
        ax.plot(x, y, label=label, color=color)

    ax.set_title(title)
    ax.legend(loc='upper left')

    # log함수 적용
    # ax.set_yscale('log')
    ax.set_xticks(x[::len(x)//11])
    ax.set_xticklabels(x_labels[::len(x_labels)//11], rotation=45)

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

x_labels = covid['자치구 기준일'].unique()
x = np.arange(len(x_labels))

plot_subplot(ax1, x, covid['전국 추가 확진'], '전국 추가 확진자', '전국 추가 확진자', 'bar', 'r')
plot_subplot(ax2, x, covid['전국 확진'], '전국 확진자', '전국 확진자', 'line', 'r')
plot_subplot(ax3, x, covid['서울시 추가 확진'], '서울시 추가 확진자', '서울시 추가 확진자', 'bar', 'b')
plot_subplot(ax4, x, covid['서울시 확진자'], '서울시 확진자', '서울시 확진자', 'line', 'b')
plot_subplot(ax5, x, covid['용산구 추가'], '용산구 추가 확진자', '용산구 추가 확진자', 'bar', 'g')
plot_subplot(ax6, x, covid['용산구 전체'], '용산구 확진자 추이', '용산구 확진자', 'line', 'g')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
