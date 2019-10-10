

from fbprophet import Prophet
import pandas as pd
import datetime

holidays=pd.read_excel('holiday.xlsx')   #读取节假日信息

turnovers_data = pd.read_excel( 'onedata.xlsx')  # 读取营业额数据

date_info = pd.read_excel('date_info.xlsx')  # 读取日期数据

turnovers_data['y']=turnovers_data['turnovers']/turnovers_data['shop_cnt']  #计算单店营业额

df=turnovers_data[['business_date','y']]    #取得需要的数据

df.rename(columns={'business_date':'ds'}, inplace = True)   #日期列列名必须是ds

rq=[datetime.datetime(2019,9,1),datetime.datetime(2019,9,30)]

df1=pd.merge(df,date_info,how='left',left_on ='ds',right_on='date')

def is_holiday(x):
#     return (x['vacation_adult']==1 or x['day_after_vacation']==1 or x['day_before_vacation']==1)
#     return (x['vacation_adult']==1 or x['day_before_vacation']==1)
    return (x['vacation_adult']==1)

df1['on_holiday']=df1.apply(is_holiday,axis=1)
df1['off_holiday']=~df1.apply(is_holiday,axis=1)

df2=df1[['ds','y','on_holiday','off_holiday']]

m = Prophet(holidays=holidays
            ,weekly_seasonality=False
            # ,yearly_seasonality=False
#             ,changepoint_prior_scale=0.01
#             ,holidays_prior_scale=1000
#             ,seasonality_prior_scale=0.01
#             ,seasonality_mode='multiplicative'
           )   #实例化，引入节假日
m.add_seasonality(name='weekly_off_holiday', period=7, fourier_order=3, condition_name='off_holiday')
# m.add_seasonality(name='yearly_off_holiday', period=365, fourier_order=10, condition_name='off_holiday')

m.fit(
    df2[
        df2['ds']<=datetime.datetime(2019,8,31)
      ]
) #建模

future = m.make_future_dataframe(periods=45)    #预测60天数据

future1=pd.merge(future,date_info,how='left',left_on='ds',right_on='date')

future1['on_holiday']=future1.apply(is_holiday,axis=1)
future1['off_holiday']=~future1.apply(is_holiday,axis=1)

future=future1[['ds','on_holiday','off_holiday']]

forecast = m.predict(future)    #预测结果

print(df[(df['ds']>=rq[0])&(df['ds']<=rq[1])]['y'].corr(forecast[(forecast['ds']>=rq[0])&(forecast['ds']<=rq[1])]['yhat']))   #相关系数
# print(forecast[forecast['ds']==datetime.datetime(2019,10,7)]['yhat'].values)
# forecast[forecast['ds']==datetime.datetime(2019,10,7)]

# fig = m.plot_components(forecast)   #观察时间序列拆分图表

# fig = m.plot(forecast)  #时间序列图

'''
#单看节假日影响
from fbprophet.plot import plot_forecast_component
plot_forecast_component(m
#                         ,forecast
                        ,forecast[(forecast['ds']>=rq[0])&(forecast['ds']<=rq[1])]
                        , 'holiday_101')
'''

result=pd.merge(df,forecast[['ds','yhat','yhat_lower','yhat_upper']],how='right',left_on='ds',right_on='ds')    #筛选输出数据

result.to_excel('prophet_pred_reuslt.xlsx',index=False)  #输出结果
