from app.data import data_info
from app.data_prepross import preprocess
from app.analizer import P_corr, splitdata,\
      decision_tree, random_forest,naive_bayes,\
      suppor_vector_machine, K_N_N

dataset = data_info()
X,y = preprocess(dataset)
P_corr(X)
X_train, X_test, y_train, y_test = splitdata(X,y)

models = [decision_tree,random_forest,naive_bayes,suppor_vector_machine,K_N_N]

for model in models:
    model(X_train,y_train,X_test,y_test)