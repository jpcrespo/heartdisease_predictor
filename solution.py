import pandas as pd
from app.data import data_info
from app.data_prepross import preprocess
from app.analizer import P_corr, decision_tree, \
      random_forest, suppor_vector_machine,\
      naive_bayes, KNN, xgboost, benchmark,benchmarkbar,\
      stackmodel


dataset = data_info()
X,y = preprocess(dataset)
P_corr(X)

models = [decision_tree,random_forest,naive_bayes,suppor_vector_machine,KNN,xgboost,stackmodel]
name_models = ["decision_tree","random_forest","naive_bayes","suppor_vector_machine","KNN","XGboost","Hybrid Model"]



accuracy=[]
precision=[]
recall=[]
f1=[]

for model in models:
      s,p,r,f = model(X,y)
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
benchmarkbar(bench_data)
