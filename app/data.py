import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def data_info():
    #the data of heart disease
    data_hs= pd.read_csv("dataset.csv")


    with open('output/dataset_info.txt', 'w') as f:
        f.write("\n*******************************************************************\n")
        f.write("The head (first five entrys) from dataset.csv:\n\n")
        f.write(data_hs.head().to_string())
        f.write("\n\n\n*******************************************************************\n")
        f.write("Resume the dataset information:\n\n")
        data_hs.info(buf=f)
        target_count = data_hs["target"].value_counts()
        f.write("\n\n*******************************************************************\n")
        f.write("Target Variable Count:\n" + "Positive - "+ str(target_count[0])+"\nNegative - "+str(target_count[1]))
        f.write("\n\n*******************************************************************\n")
        f.write("Number of data entrys:\n"+str(data_hs.shape[0]))
        f.write("\nNumber of independent variables:\n"+str(data_hs.shape[1]-1))
    sns.countplot(data=data_hs, x='target')
    plt.title("Count of Positive/Negative Target")
    plt.xticks(ticks=[0, 1], labels=['Negative '+str(target_count[0]), 'Positive '+ str(target_count[1])])
    plt.grid()
    plt.savefig('output/countplot_target.png', dpi=300)
    return data_hs