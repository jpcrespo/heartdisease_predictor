"""
This module containe distinct algorithms
from ML and anothers to show results
"""


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score,precision_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#Using Pearson Correlation
def P_corr(X):
    plt.figure(figsize=(12,10))
    cor = X.corr()
    sns.heatmap(cor, annot=True, cmap='Accent')
    plt.title("Pearson Correlation",fontsize=30)
    plt.savefig('output/Pearson_corr.png', dpi=300)

def splitdata(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)  
    return X_train, X_test, y_train, y_test


def decision_tree(X_train,y_train,X_test,y_test):
    DT = DecisionTreeClassifier(max_depth = 40)
    model = DT.fit(X_train,y_train)
    aucc = model.score(X_train, y_train)
    y_pred = DT.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    with open('output/decision_tree.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Decision Tree algorithm\n")
        f.write('\nPrecission training: '+str(aucc))
        f.write('\nAccuracy testing: '+str(acc))
        f.write('\nPrecission testing: '+str(prec))
        f.write('\nRecall training: '+str(rec))
        f.write('\nf1 training: '+str(f1))
    ConfusionMatrixDisplay.from_estimator(DT, X_test, y_test)
    plt.title('Confusion Matrix\nDecision Tree',fontsize=18)
    plt.savefig('output/DT.png', dpi=300)
 


def random_forest(X_train, y_train, X_test, y_test):
    RF = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42)
    model = RF.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    y_pred = RF.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/random_forest.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {train_accuracy:.4f}')
        f.write(f'\nAccuracy testing: {test_accuracy:.4f}')
        f.write(f'\nPrecision testing: {precision:.4f}')
        f.write(f'\nRecall training: {recall:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(RF, X_test, y_test)
    plt.title('Confusion Matrix\nRandom Forest',fontsize=18)
    plt.savefig('output/RF.png', dpi=300)


def suppor_vector_machine(X_train, y_train, X_test, y_test):
    svc = SVC(gamma='auto')
    model = svc.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    y_pred = svc.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/suppor_vector_machine.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {train_accuracy:.4f}')
        f.write(f'\nAccuracy testing: {test_accuracy:.4f}')
        f.write(f'\nPrecision testing: {precision:.4f}')
        f.write(f'\nRecall training: {recall:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test)
    plt.title('Confusion Matrix\nSupport Vector Machine',fontsize=18)
    plt.savefig('output/SVC.png', dpi=300)


def naive_bayes(X_train, y_train, X_test, y_test):
    nb = GaussianNB()
    model = nb.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    y_pred = nb.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/naive_bayes.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {train_accuracy:.4f}')
        f.write(f'\nAccuracy testing: {test_accuracy:.4f}')
        f.write(f'\nPrecision testing: {precision:.4f}')
        f.write(f'\nRecall training: {recall:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(nb, X_test, y_test)
    plt.title('Confusion Matrix\nNaive Bayes',fontsize=18)
    plt.savefig('output/NB.png', dpi=300)



def K_N_N(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    model = knn.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    y_pred = knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/KNN.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {train_accuracy:.4f}')
        f.write(f'\nAccuracy testing: {test_accuracy:.4f}')
        f.write(f'\nPrecision testing: {precision:.4f}')
        f.write(f'\nRecall training: {recall:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
    plt.title('Confusion Matrix\nKNN',fontsize=18)
    plt.savefig('output/knn.png', dpi=300)


    