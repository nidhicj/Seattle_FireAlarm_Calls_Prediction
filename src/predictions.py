import numpy as np
import pandas as pd
import xgboost as xgb
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def truth_vs_predictions(home_folder,predictions, ground_truth, time = ''):
    fig,ax = plt.subplots(figsize = [15,5])
    
    if time == 'weekly':
        predictions.loc[(predictions.index > '2022-08-01') & (predictions.index < '2022-08-08')]['prediction'].plot()
        ground_truth.loc[(ground_truth.index > '2022-08-01') & (ground_truth.index < '2022-08-08')]['NUM_CALLS'].plot()
        plt.legend(['Truth Data','Prediction'],loc='upper left')
        plt.title('Predictions vs Ground Truth for a Week')
        plt.xlabel('Time')
        plt.ylabel('Number of calls')
        plt.savefig(home_folder+"/figures/predictions_comparison/weekly_plots.png")
        plt.close()

    elif time == 'monthly':
        predictions.loc[(predictions.index > '2022-07-01') & (predictions.index < '2022-08-01')]['prediction'].plot(color = 'green')
        ground_truth.loc[(ground_truth.index > '2022-07-01') & (ground_truth.index < '2022-08-01')]['NUM_CALLS'].plot(color = 'orange')
        plt.legend(['Truth Data','Prediction'],loc='upper left')
        plt.title('Predictions vs Ground Truth for a Week')
        plt.xlabel('Time')
        plt.ylabel('Number of calls')
        plt.savefig(home_folder+"/figures/predictions_comparison/monthly_plot.png")
        plt.close()

    else:
        #truth vs predictions for entire dataset
        ground_truth[['NUM_CALLS']].plot.line(figsize=(15, 5))
        predictions['prediction'].plot.line()
        ax.set_title('Predictions and Ground Truth')
        plt.xlabel('Time')
        plt.ylabel('Number of calls')
        plt.legend(loc='upper left')
        plt.savefig(home_folder+"/figures/predictions_comparison/Truth_vs_Prediction.png")
        plt.close()



def model_performance(X_train, y_train,X_test,target,test,home_folder):
    reg = reg_model(X_train, y_train)
    reg.load_model(home_folder+"/models/regression_model.json")

    #Test model performance in comparison with actual data (backtesting)

    #For all the 5 years training data, tested for this year's data (2022)
    test['prediction'] = reg.predict(X_test)
    preds_truth = target.merge(test[['prediction']], how='left', left_index=True, right_index=True)

    predictions = pd.DataFrame(preds_truth.loc[:,'prediction'])
    ground_truth = pd.DataFrame(preds_truth.loc[:,'NUM_CALLS'])
    
    truth_vs_predictions(home_folder, predictions, ground_truth, time = 'weekly')
    truth_vs_predictions(home_folder, predictions, ground_truth, time = 'monthly')
    truth_vs_predictions(home_folder, predictions, ground_truth)

    score = np.sqrt(mean_squared_error(test['NUM_CALLS'], test['prediction']))
    print(f'RMSE Score on Test set: {score:0.2f}')


