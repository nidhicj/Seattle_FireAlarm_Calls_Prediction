import numpy as np
import pandas as pd
import argparse
from src.model import *
import unittest as ut

odd_months = [1,3,5,7,8,10,12]
even_months = [4,6,9,11]

def input_mods(args):
    all_user_inputs = []
    for j in range(len(args.datetime)):
        new_input = np.zeros(7)
        for i in range(len(new_input)):
            if i == len(args.datetime[j]):
                break
            else:
                new_input[i] = args.datetime[j][i]
        all_user_inputs.append(new_input)
    return np.array(all_user_inputs)


    
def test_wrong_dates(all_user_inputs):
    
    for date in all_user_inputs:
        if (int(date[1]) in odd_months) and (1 <= date[2] <=31):
            pass
        elif (int(date[1]) in even_months) and (1 <= date[2] <=30):
            pass
        elif (int(date[1])== 2) and (1 <= date[2] <=28) and (date[0]%4!=0):
            pass
        elif (int(date[1])== 2) and (1 <= date[2] <=29) and (date[0]%4==0):
            pass
        else:
            raise Exception("Sorry, the date is not correct") 
    for time in all_user_inputs:
        assert (0 <= int(time[3]) <=23 ) and (0 <= time[4] <=59)
        
    #for better accuracy conditions the testing timeline is set for dates after 2017-01-01
    for border_time in all_user_inputs:
        assert (int(border_time[1]) >= 1) and (border_time[2] >=1) and (border_time[0] >=2017)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-dt','--datetime', type=float, nargs='+', action='append', help='specify datetime in Y M D Hr Min Sec Temp Windspeed')
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
    test_wrong_dates(all_user_inputs)
    new_output = abs(reg.predict(all_user_inputs).astype(int))
    print("The predicted number of calls for these dates are:")
    print(new_output)



