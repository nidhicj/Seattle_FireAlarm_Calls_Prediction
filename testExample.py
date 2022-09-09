import numpy as np
import pandas as pd
import argparse
from src.model import *

def input_mods(args):
    all_user_inputs = []
    for j in range(len(args.datetime)):
        user_input = np.zeros(7)
        for i in range(len(user_input)):

            if args.datetime[j][i] == args.datetime[j][-1]:
                break
            else:
                user_input[i] = args.datetime[j][i]
        all_user_inputs.append(user_input)
    return np.array(all_user_inputs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-dt','--datetime', type=int, nargs='+', action='append', 
    help='file list')
    args = parser.parse_args()
    
    #Initializaing datasets and folders
    home_folder = "/mnt/d/Germany/folder_01/Seattle_Calls_Prediction" 
    target = pd.read_csv(home_folder+"/data/meta_data.csv")
    target = target.set_index("Datetime")
    train = target.loc[target.index < '2022-01-01']
    test = target.loc[target.index >= '2022-01-01']

    #Initializaing features and train/test sets
    list_of_features = ['year','month','day','hour','minutes','temp','wspd']
    num_calls = 'NUM_CALLS'
    X_train, y_train = train[list_of_features], train[num_calls]
    X_test, y_test  = test[list_of_features], test[num_calls]

    #Initializaing model and loading the trained one
    reg = reg_model(X_train, y_train,hype=False)
    reg.load_model(home_folder+"/models/regression_model.json")
    
    #Working with the user input for 
    all_user_inputs = input_mods(args)

    new_output = reg.predict(all_user_inputs).astype(int)
    print("The predicted number of calls for these dates are:")
    print(new_output)



