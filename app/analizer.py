"""
This module containe distinct algorithms
from ML and anothers to show results
"""


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score,precision_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

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
    with open('output/DT.txt', 'w') as f:
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
    return acc,prec,rec,f1


def random_forest(X_train, y_train, X_test, y_test):
    RF = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42)
    model = RF.fit(X_train, y_train)
    aucc = model.score(X_train, y_train)
    y_pred = RF.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/random_forest.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {aucc:.4f}')
        f.write(f'\nAccuracy testing: {acc:.4f}')
        f.write(f'\nPrecision testing: {prec:.4f}')
        f.write(f'\nRecall training: {rec:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(RF, X_test, y_test)
    plt.title('Confusion Matrix\nRandom Forest',fontsize=18)
    plt.savefig('output/RF.png', dpi=300)
    return acc,prec,rec,f1

def suppor_vector_machine(X_train, y_train, X_test, y_test):
    svc = SVC(gamma='auto')
    model = svc.fit(X_train, y_train)
    aucc = model.score(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/suppor_vector_machine.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {aucc:.4f}')
        f.write(f'\nAccuracy testing: {acc:.4f}')
        f.write(f'\nPrecision testing: {prec:.4f}')
        f.write(f'\nRecall training: {rec:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test)
    plt.title('Confusion Matrix\nSupport Vector Machine',fontsize=18)
    plt.savefig('output/SVC.png', dpi=300)
    return acc,prec,rec,f1

def naive_bayes(X_train, y_train, X_test, y_test):
    nb = GaussianNB()
    model = nb.fit(X_train, y_train)
    aucc = model.score(X_train, y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/naive_bayes.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {aucc:.4f}')
        f.write(f'\nAccuracy testing: {acc:.4f}')
        f.write(f'\nPrecision testing: {prec:.4f}')
        f.write(f'\nRecall training: {rec:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(nb, X_test, y_test)
    plt.title('Confusion Matrix\nNaive Bayes',fontsize=18)
    plt.savefig('output/NB.png', dpi=300)
    return acc,prec,rec,f1


def xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier()
    model = xgb.fit(X_train, y_train)
    aucc = model.score(X_train, y_train)
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/xgboost.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {aucc:.4f}')
        f.write(f'\nAccuracy testing: {acc:.4f}')
        f.write(f'\nPrecision testing: {prec:.4f}')
        f.write(f'\nRecall training: {rec:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test)
    plt.title('Confusion Matrix\nXGBOOST',fontsize=18)
    plt.savefig('output/xgboost.png', dpi=300)
    return acc,prec,rec,f1
    


def K_N_N(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    model = knn.fit(X_train, y_train)
    aucc = model.score(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    with open('output/KNN.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("The Scores for Random Forest algorithm\n")
        f.write(f'\nPrecision training: {aucc:.4f}')
        f.write(f'\nAccuracy testing: {acc:.4f}')
        f.write(f'\nPrecision testing: {prec:.4f}')
        f.write(f'\nRecall training: {rec:.4f}')
        f.write(f'\nF1 training: {f1:.4f}')
    ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
    plt.title('Confusion Matrix\nKNN',fontsize=18)
    plt.savefig('output/knn.png', dpi=300)
    return acc,prec,rec,f1
    









def benchmarkbar():
    # Creación del DataFrame
    data = {
        'Model': ['decision_tree', 'random_forest', 'naive_bayes', 'support_vector_machine', 'K_N_N','XgBoost'],
        'Accuracy': [0.670330, 0.868132, 0.846154, 0.857143, 0.846154,0.7253],
        'Precision': [0.720930, 0.877551, 0.906977, 0.860000, 0.857143,0.7963],
        'Recall': [0.632653, 0.877551, 0.795918, 0.877551, 0.857143,0.7544],
        'F1 Score': [0.673913, 0.877551, 0.848726, 0.868687, 0.857143,0.7748]
    }

    bench_data = pd.DataFrame(data)

    # Transformar los datos para que se ajusten al formato adecuado para seaborn
    melted_data = bench_data.melt(id_vars='Model', var_name='Metric', value_name='Score')

    # Crear un gráfico de barras
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño para mejor visualización
    barplot = sns.barplot(data=melted_data, x='Metric', y='Score', hue='Model', palette='plasma')

    # Mejoras estéticas
    plt.title('Benchmark', fontsize=16)  # Título del gráfico
    plt.xlabel('Metric', fontsize=14)  # Etiqueta del eje X
    plt.ylabel('Score', fontsize=14)  # Etiqueta del eje Y
    plt.ylim(0.6, 0.9)  # Ajustar los límites del eje Y para mejor enfoque

    # Ajustar la leyenda
    plt.legend(title='ML Model', fontsize=12, title_fontsize='13', loc='upper left', bbox_to_anchor=(1, 1))  # Posicionamiento de la leyenda

    # Ajustar el layout para evitar recortes y superposiciones
    plt.tight_layout()

    # Mostrar el gráfico
    plt.savefig('output/barbench.png', dpi=300)


def benchmark(bench_data):
    categories = list(bench_data)[1:]  # Esto excluye la primera columna que es 'Model'
    N = len(categories)

    angles = [n/float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    fig, ax = plt.subplots(figsize=(20,15), subplot_kw={'polar': True})
    plt.xticks(angles[:-1],categories, color='b',size=18)
    ax.set_rlabel_position(270)

    for i in range(bench_data.shape[0]):
        values = bench_data.loc[i].drop('Model').values.flatten().tolist()
        values += values[:1]  # Se completa el círculo
        ax.plot(angles,np.log(values), linewidth=2, linestyle='--', label=bench_data['Model'][i])
        ax.fill(angles,np.log(values), alpha=0.05)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if label.get_text() == 'Accuracy' or label.get_text() == 'Recall' :
            label.set_horizontalalignment('left' if angle < np.pi else 'right')
            label.set_verticalalignment('bottom' if angle < np.pi/2 or angle > 3*np.pi/2 else 'top')

    ax.set_yticklabels([])
    plt.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(0.15, 0.15),fontsize=16)
    plt.savefig('output/bench.png', dpi=300)
