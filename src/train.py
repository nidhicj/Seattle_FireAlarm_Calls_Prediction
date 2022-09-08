
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



target = pd.read_csv("./data/meta_data.csv")
train = target.loc[target.index < '2022-01-01']
test = target.loc[target.index >= '2022-01-01']

FEATURES = ['hour',	'dayoftheweek','month','dayofyear','seasonofyear','temp','wspd']
TARGET = 'target'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

#Tuning the model
"""
dfdf
"""

# reg = xgb.XGBRegressor(n_estimators = 1000)
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=20,
                       objective='reg:linear',
                       max_depth=4,
                       learning_rate=0.99)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

#Test model performance in comparison with actual data (backtesting)

#For all the 5 years training data, tested for this year's data (2022)
test['prediction'] = reg.predict(X_test)
target_test = target.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = target[['target']].plot.line(figsize=(15, 5))
target_test['prediction'].plot.line(ax=ax)
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

#Compare weekly data with truth and predictions 
ax = target_test.loc[(target_test.index > '04-01-2022') & (target_test.index < '04-08-2022')]['target'] \
    .plot(figsize=(15, 5), title='Week Of Data')
target_test.loc[(target_test.index > '04-01-2022') & (target_test.index < '04-08-2022')]['prediction'] \
    .plot.line()
plt.legend(['Truth Data','Prediction'])
plt.show()

#Compare monthly data with truth and predictions 
ax = target_test.loc[(target_test.index > '04-01-2022') & (target_test.index < '05-01-2022')]['target'] \
    .plot(figsize=(15, 5), title='Week Of Data')
target_test.loc[(target_test.index > '04-01-2022') & (target_test.index < '05-01-2022')]['prediction'] \
    .plot.line()
plt.legend(['Truth Data','Prediction'])
plt.show()

#Checking the RMSE on test data

score = np.sqrt(mean_squared_error(test['target'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

