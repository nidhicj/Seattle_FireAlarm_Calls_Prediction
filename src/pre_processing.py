import os 
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from google.cloud import bigquery
from meteostat import Point, Hourly
from google.oauth2 import service_account


def call_per_time(home_folder,target,timeline):
  _ , ax = plt.subplots(figsize = (15,5))
  sns.boxplot(data = target, x = timeline, y="NUM_CALLS")
  ax.set_title("calls per "+timeline)
  plt.savefig(home_folder+"/figures/pre_processing/call_per_"+timeline+".png")

def pre_processing(home_folder):
#initalize variables for SQL in Bigquery
  
  df = pd.read_csv('https://data.seattle.gov/api/views/kzjm-xkqj/rows.csv?accessType=DOWNLOAD')
  df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%Y %I:%M:%S %p")
  df_meta = df.drop(columns = [ "Address","Report Location","Incident Number"], inplace = False).sort_values(by="Datetime",ascending=True)
  df_alpha = df_meta[(df_meta['Datetime'] > '2017-01-01') & (df_meta['Datetime'] <= '2022-09-02')]
  target = df_alpha.groupby([pd.Grouper(key='Datetime',freq='H')]).size().reset_index(name='NUM_CALLS')
  target = target.set_index("Datetime")

  # Visualize training and test sets
  train = target.loc[target.index < '2022-01-01']
  test = target.loc[target.index >= '2022-01-01']
  _, az = plt.subplots(figsize=(15,5))
  train.plot(ax = az,label = "tr")
  test.plot(ax = az,label = "te")
  az.axvline('01-01-2022', color='black', ls='--')
  az.legend(['Training Set', 'Test Set'])
  plt.savefig(home_folder+'/figures/pre_processing/Training_Test_Sets.png')

  #Segregate time elements as columns
  target["year"] = target.index.year
  target["month"] = target.index.month
  target["day"] = target.index.day
  target["hour"] = target.index.hour
  target["minutes"] = target.index.minute

  #Adding weather data according to Seattle location on hourly basis

  start = datetime(2017, 1, 1,0,0,0)
  end = datetime(2022, 9, 1,23,59,59)
  seattle = Point(47.6062, -122.3321)
  hourly = Hourly(seattle, start, end)
  hourly = hourly.fetch()
  target["temp"] = hourly.loc[:]["temp"]
  target["wspd"] = hourly.loc[:]["wspd"]


  #Visualize calls per hour/month/year
  call_per_time(home_folder,target,'hour')
  call_per_time(home_folder,target,'month')
  call_per_time(home_folder,target,'year')
  target.to_csv(home_folder+"/data/meta_data.csv")
