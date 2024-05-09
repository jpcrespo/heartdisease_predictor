from app.data import data_info
from app.data_prepross import preprocess
from app.analizer import P_corr
dataset = data_info()
X,y = preprocess(dataset)
P_corr(X)
