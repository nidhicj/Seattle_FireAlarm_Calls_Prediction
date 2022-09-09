from pre_processing import *
from train import *
from predictions import *

if __name__ == "__main__" :
    home_folder = "<<Initialize your home_folder destination>>"
    
    print("Carrying out Feature Engineering")
    pre_processing(home_folder)

    target = pd.read_csv(home_folder+"/data/meta_data.csv")
    target = target.set_index("Datetime")
    train = target.loc[target.index < '2022-01-01']
    test = target.loc[target.index >= '2022-01-01']

    list_of_features = ['year','month','day','hour','minutes','temp','wspd']
    num_calls = 'NUM_CALLS'

    X_train, y_train = train[list_of_features], train[num_calls]
    X_test, y_test  = test[list_of_features], test[num_calls]
    
    print("Training the dataset")
    training_model(X_train, y_train,X_test, y_test,home_folder,hype=False) 
    #turn hype=True if hyperparameter tuning is to be done
    #one such iteration has been already carried out and best_estimated model is initilazed
    
    print("Evaluating the model performance")
    model_performance(X_train, y_train,X_test,target,test,home_folder)
    