

from fbprophet import Prophet
import pandas as pd
import datetime

holidays=pd.read_excel('holiday.xlsx')   #读取节假日信息

turnovers_data = pd.read_excel( 'onedata.xlsx')  # 读取营业额数据

turnovers_data['y']=turnovers_data['turnovers']/turnovers_data['shop_cnt']  #计算单店营业额

df=turnovers_data[['business_date','y']]    #取得需要的数据

df.rename(columns={'business_date':'ds'}, inplace = True)   #日期列列名必须是ds

rq=[datetime.datetime(2019,9,25),datetime.datetime(2019,10,7)]

m = Prophet(holidays=holidays
#             ,weekly_seasonality=True
#             ,changepoint_prior_scale=0.01
#             ,holidays_prior_scale=1000
#             ,seasonality_prior_scale=0.01
#             ,seasonality_mode='multiplicative'
           )   #实例化，引入节假日

# m.add_seasonality(    name='weekly', period=7, fourier_order=3, prior_scale=10)

m.fit(
    df[
        df['ds']<datetime.datetime(2019,9,15)
      ]
) #建模

future = m.make_future_dataframe(periods=45)    #预测60天数据
forecast = m.predict(future)    #预测结果

# print(df[(df['ds']>=rq[0])&(df['ds']<=rq[1])]['y'].corr(forecast[(forecast['ds']>=rq[0])&(forecast['ds']<=rq[1])]['yhat']))   #相关系数

# print(forecast[forecast['ds']==datetime.datetime(2019,10,7)]['yhat'].values)
# forecast[forecast['ds']==datetime.datetime(2019,10,7)]

# fig = m.plot_components(forecast)   #观察时间序列拆分图表

# fig = m.plot(forecast)  #时间序列图

#单看节假日影响
from fbprophet.plot import plot_forecast_component
plot_forecast_component(m
#                         ,forecast
                        ,forecast[(forecast['ds']>=rq[0])&(forecast['ds']<=rq[1])]
                        , 'holiday_101')


result=pd.merge(df,forecast[['ds','yhat','yhat_lower','yhat_upper']],how='right',left_on='ds',right_on='ds')    #筛选输出数据

result.to_excel('prophet_pred_reuslt.xlsx',index=False)  #输出结果
