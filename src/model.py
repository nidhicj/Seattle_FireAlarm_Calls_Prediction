import time
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

#Tuning the model
def tuning_model(X_train, y_train):
    # A parameter grid for XGBoost
    params = {
    'n_estimators':[500,1000,1500,2000],
    'objective': ['reg:squarederror', 'reg:tweedie'],
    'booster': ['gbtree', 'gblinear'],
    'eval_metric': ['rmse'],
    'eta': [i/10.0 for i in range(3,6)],
    }

    reg = XGBRegressor()

    # run randomized search
    n_iter_search = 10
    random_search = RandomizedSearchCV(reg, param_distributions=params,
                                    n_iter=n_iter_search)

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

#Model training after best predictor parameters from tuning
def reg_model(X_train, y_train,hype=False):
    if hype == True:
        reg = tuning_model(X_train, y_train)
    else:
        #the model below is obtained after the hyperparameter tuning is done
        #this initialization is for simplicity
        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                    eta=0.4, eval_metric='rmse', gamma=0, gpu_id=-1,
                    importance_type=None, interaction_constraints='',
                    learning_rate=0.900000006, max_delta_step=0, max_depth=6,
                    min_child_weight=1, monotone_constraints='()',
                    n_estimators=1500, n_jobs=8, num_parallel_tree=1, predictor='auto',
                    random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                    subsample=1, tree_method='exact', validate_parameters=1,
                    verbosity=None)
    return reg