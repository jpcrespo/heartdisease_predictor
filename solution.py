from app.data import data_info
from app.data_prepross import preprocess
from app.analizer import P_corr, splitdata, decision_tree

dataset = data_info()
X,y = preprocess(dataset)
P_corr(X)
X_train, X_test, y_train, y_test = splitdata(X,y)
decision_tree(X_train,y_train,X_test,y_test)
