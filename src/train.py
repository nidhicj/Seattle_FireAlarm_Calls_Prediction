import pandas as pd
from model import *
from xgboost import plot_importance
import matplotlib.pyplot as plt

def training_model(X_train, y_train,X_test, y_test,home_folder,hype):
        reg = reg_model(X_train, y_train, hype=False) #hype=True if hyperparameter tuning is needed
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)
        plot_importance(reg)
        plt.savefig(home_folder+"/figures/feature_importance/importance.png")
        reg.save_model(home_folder+"/models/regression_model.json")


