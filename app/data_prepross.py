from sklearn import preprocessing
import pandas as pd

def preprocess(dataframe):

    df = dataframe.copy()

    # take independant variables
    df = df.drop(['target'], axis=1)
    r_scaler = preprocessing.MinMaxScaler()
    r_scaler.fit(df)
    # scale the dataset
    modified_data = pd.DataFrame(r_scaler.transform(df), index=df.index, columns=df.columns)
    X = modified_data
    y = dataframe['target']
    with open('output/dataset_info.txt', 'a') as f:
        f.write("\n\n\n*******************************************************************\n")
        f.write("The new head (first five entrys) normalized:\n\n")
        f.write(X.head().to_string())
    return X,y
    