import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Hourly

# df = pd.read_csv('https://data.seattle.gov/api/views/kzjm-xkqj/rows.csv?accessType=DOWNLOAD')
# df["Datetime"] = pd.to_datetime(df["Datetime"], format="%m/%d/%Y %I:%M:%S %p")
# df_meta = df.drop(columns = [ "Address","Report Location","Incident Number"], inplace = False).sort_values(by="Datetime",ascending=True)
# df_alpha = df_meta[(df_meta['Datetime'] >= '2017-01-01') ]
# target = df_alpha.groupby([pd.Grouper(key='Datetime',freq='H')]).size().reset_index(name='target')
# target = target.set_index("Datetime")

from google.cloud import bigquery
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('/mnt/d/Germany/JobHunt/Niologic/warm-bonfire-315806-57e1dcb7bd95.json')
project_id = 'warm-bonfire-315806'
client = bigquery.Client(credentials=credentials, project=project_id)
import os 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./warm-bonfire-315806-57e1dcb7bd95.json"

query_job = client.query(""" 
SELECT 
  datetime_trunc(Datetime, HOUR) as time,
  COUNT(*) AS NUM_CALLS
FROM `warm-bonfire-315806.seattle_call_volume.call_freq_prediction` 
WHERE Datetime >='2017-01-01'
GROUP BY time
ORDER BY time
""")
target = query_job.result().to_dataframe()
target = target.set_index("time")

# Visualize training and test sets
color_pal = sns.color_palette()
train = target.loc[target.index < '2022-01-01']
test = target.loc[target.index >= '2022-01-01']
fig, az = plt.subplots(figsize=(15,5))
train.plot(ax = az,label = "tr")
test.plot(ax = az,label = "te")
az.axvline('01-01-2022', color='black', ls='--')
az.legend(['Training Set', 'Test Set'])
plt.savefig("/mnt/d/Germany/JobHunt/Niologic/results/Training_Test_Sets.png")

#Segregate time elements as columns
target["year"] = target.index.year
target["month"] = target.index.month
target["day"] = target.index.day
target["hour"] = target.index.hour
target["minutes"] = target.index.minute

#Visualize calls per hour/month/year
fig , ax = plt.subplots(figsize = (15,5))

sns.boxplot(data = target, x = "hour", y="NUM_CALLS")
ax.set_title("calls per hour")
plt.savefig("/mnt/d/Germany/JobHunt/Niologic/results/call_per_hour")

sns.boxplot(data = target, x = "month", y = "NUM_CALLS")
ax.set_title("calls per month")
plt.savefig("/mnt/d/Germany/JobHunt/Niologic/results/call_per_month")

sns.boxplot(data = target, x = "year", y = "NUM_CALLS")
ax.set_title("calls per year")
plt.savefig("/mnt/d/Germany/JobHunt/Niologic/results/call_per_year")

#Adding weather data according to Seattle location on hourly basis
# Set time period
start = datetime(2017, 1, 1,0,0,0)
end = datetime(2022, 9, 1,23,59,59)

# Create Point for Seattle
seattle = Point(47.6062, -122.3321)

# Get daily data for 2018
data = Hourly(seattle, start, end)
data = data.fetch()
target["temp"] = data.loc[:]["temp"]
target["wspd"] = data.loc[:]["wspd"]

target.to_csv("/mnt/d/Germany/JobHunt/Niologic/data/meta_data.csv")