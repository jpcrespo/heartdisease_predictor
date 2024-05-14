"""
This module containe distinct algorithms
from ML and anothers to show results
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Metrics and validation methods
from sklearn.model_selection import cross_validate

# Machine Learning Methods
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'}

#Using Pearson Correlation
def P_corr(X):
    plt.figure(figsize=(12,10))
    cor = X.corr()
    sns.heatmap(cor, annot=True, cmap='Accent')
    plt.title("Pearson Correlation",fontsize=30)
    plt.savefig('output/Pearson_corr.png', dpi=300)

def decision_tree(X,y):
    DT = DecisionTreeClassifier(max_depth = 40)

    cv_results = cross_validate(DT, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()


def random_forest(X,y):
    RF = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42)
    cv_results = cross_validate(RF, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()



def suppor_vector_machine(X,y):
    svc = SVC(gamma='auto')
    cv_results = cross_validate(svc, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()

def naive_bayes(X,y):
    nb = GaussianNB()
    cv_results = cross_validate(nb, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()


def KNN(X,y):
    knn = KNeighborsClassifier(n_neighbors=3)
    cv_results = cross_validate(knn, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()




def xgboost(X,y):
    xgb = XGBClassifier()
    cv_results = cross_validate(xgb, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()

def benchmarkbar(bench_data):
    # Transformar los datos para que se ajusten al formato adecuado para seaborn
    melted_data = bench_data.melt(id_vars='Model', var_name='Metric', value_name='Score')

    # Crear un gráfico de barras
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño para mejor visualización
    barplot = sns.barplot(data=melted_data, x='Metric', y='Score', hue='Model', palette='plasma')

    # Mejoras estéticas
    plt.title('Benchmark', fontsize=16)  # Título del gráfico
    plt.xlabel('Metric', fontsize=14)  # Etiqueta del eje X
    plt.ylabel('Score', fontsize=14)  # Etiqueta del eje Y
    plt.ylim(0.5, 1)  # Ajustar los límites del eje Y para mejor enfoque

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
    with open('output/benchmark_results.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("Benchmark Results\n")
        f.write("*******************************************************************\n\n")
        f.write(bench_data.to_string(index=False))
