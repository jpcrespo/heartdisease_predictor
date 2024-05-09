import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score, ConfusionMatrixDisplay


#Using Pearson Correlation
def P_corr(X):
    plt.figure(figsize=(12,10))
    cor = X.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Accent)
    plt.title("Pearson Correlation",fontsize=30)
    plt.savefig('output/Pearson_corr.png', dpi=300)

def splitdata(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)  
    return X_train, X_test, y_train, y_test


def decision_tree(X_train,y_train,X_test,y_test):
    DT = tree.DecisionTreeClassifier(max_depth = 40)
    DT.fit(X_train,y_train)
    aucc = DT.score(X_train, y_train)
    y_pred = DT.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    ConfusionMatrixDisplay.from_estimator(DT, X_test, y_test)
    plt.title('Confusion Matrix\nDecision Tree',fontsize=18)
    plt.savefig('output/DT.png', dpi=300)
 