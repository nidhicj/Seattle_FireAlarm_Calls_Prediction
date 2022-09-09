# Prediction of Call Volume for Seattle Fire Department

## Objective
The following repository tackles the regression task of prediciton of call volume for Seattle Fire Department.
The Seattle Real Time Fire 911 Calls [dataset](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj) is maintained by US government for every 5 minutes. although, for this assessmnent, the past 5 years data (2017- Sept 2022) is utilized. The data hence obtained is pre-processed, trained and tested on a regression model namely XGBoost 

## Description

The source code for this assessment consists of pre-processing, models, train and prediction tasked files. For the complete analysis the following pipeline was adopted to understand the data and extract useful data

![ML Pipeline](/figures/pipeline.png "Shiprock, New Mexico by Beau Rogers")

## Pre-processing 
In this section, the data is first filtered out for past 5 years and sorted in ascending order (past date first). The number of calls occuring every hour for are segregated which forms the ground truth of the dataset. This data is visualized to check for any outliers or anomaly in the dataset. 
Subsequently, `meteostat` library is used to obtaine the hourly weather data (namely temperature and windspeed) for Seattle city (coordinated being 47.6062, -122.3321) and then added to the current dataset. 
Having obtained the raw dataset, the exploratory dataset anaysis is carried out to visualize the call frequency for every hour/month/year.
It is seen that the most influential feature is the 'hours' where the call volume is sporadically distributed over a span of 24 hours. The most number of calls occur during 10:00hrs to 18:00hrs. Moreover, the call volume every month and year stays almost the same.
As a result, the features selected for the analysis are:
'year','month','day','hour','minutes','temp','windspeed' and ground truth is 'number of calls'

In `pre_process.py` file, the aforementioned tasks are accomplished and the visulaizations associating to the analysis are saved in `figures/pre_processing` folder

## Model training and predicting

The feature engineered dataset obtained after pre-processing is utilized for training and prediction of calls. The ML model chosen for this assessment is xgboost as it is efficient and provides many scalable hyperparameters.Firstly the model is tuned to obtain the best hyperparameters using random search method. These best estimators are then initialized to fit the model on training data. Then the important features are plotted and it is seen that in datetime features, the 'hour' feature is highly influential as from the pre-processing. Although temperature is a also critical feature. 

The model is now used on the test dataset to get the predictions and the new comparison plot is visualized to check the model predictions with the ground truth. The Root Mean Square Error (RMSE) evaluation metric is used to check model performance its value obtained on the test set is 4.7

The supporting files for these tasks are in `train.py`,`predictions.py` and `model.py`. The supporting plots of comparison and feature importance are saved in `figures/feature_importance` and `figures/predictions_comparison` folders. 

## Rebuild and Testing

To rebuild this assessment, please install the dependecies or use .yml file for the similar conda environment. The `src/main.py` file can be used to rebuild the this task and a model will be trained and predicted up on. The results will be saved in respective folders.

To test the obtained model, use `testExample.py` file in terminal with the script:
`python testExamply.py -dt <datetime/weather-01> -dt <datetime/weather-01> ...`

where the you can specify 'n' number of datetime along with weather data (if available) to obtain the number of calls for the specified datetime. 
### For example
To get the call volume for the dates: <br>
2018-09-30 22:09:00 -- -- <br>
2021-06-24 10:00:8.9 -- --<br>
2022-04-03 09:25:03 4.6 <br>
`python testExample.py -dt 2018 9 30 22 9 -dt 2021 6 24 10 0 8.9 -dt 2022 4 3 9 25 3 4.6`

The output obtained is <br>
`The predicted number of calls for these dates are:` <br>
`[30 12  8]`


