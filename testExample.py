from unicodedata import name
import pandas as pd

if __name__ == '__main__':
    home_folder = "/mnt/d/Germany/folder_01/Seattle_Calls_Prediction"
    target = pd.read_csv(home_folder+"/data/meta_data.csv")
    target = target.set_index("Datetime")
    train = target.loc[target.index < '2022-01-01']
    test = target.loc[target.index >= '2022-01-01']

    list_of_features = ['year','month','day','hour','minutes','temp','wspd']
    num_calls = 'NUM_CALLS'

    X_train, y_train = train[list_of_features], train[num_calls]
    X_test, y_test  = test[list_of_features], test[num_calls]

    