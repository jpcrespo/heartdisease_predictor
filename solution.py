import pandas as pd
from app.data import data_info
from app.data_prepross import preprocess
from app.analizer import P_corr, splitdata,\
      decision_tree, random_forest,naive_bayes,\
      suppor_vector_machine, K_N_N, benchmarkbar, benchmark

dataset = data_info()
X,y = preprocess(dataset)
P_corr(X)
X_train, X_test, y_train, y_test = splitdata(X,y)

models = [decision_tree,random_forest,naive_bayes,suppor_vector_machine,K_N_N]
name_models = ["decision_tree","random_forest","naive_bayes","suppor_vector_machine","K_N_N"]

accuracy=[]
precision=[]
recall=[]
f1=[]

for model in models:
      s,p,r,f = model(X_train,y_train,X_test,y_test)
      accuracy.append(s)
      precision.append(p)
      recall.append(r)
      f1.append(f)

#Create de Benchmark data
bench_data = pd.DataFrame({
    'Model': name_models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

benchmark(bench_data)
benchmarkbar()