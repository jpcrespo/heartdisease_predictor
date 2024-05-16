"""
This module containe distinct algorithms
from ML and anothers to show results
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Metrics and validation methods
from sklearn.model_selection import cross_validate, GridSearchCV

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

def tune_hyperparameters(model, param_grid, X, y):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params

#Using Pearson Correlation
def P_corr(X):
    plt.figure(figsize=(12,10))
    cor = X.corr()
    sns.heatmap(cor, annot=True, cmap='Accent')
    plt.title("Pearson Correlation",fontsize=30)
    plt.savefig('output/Pearson_corr.png', dpi=300)

def decision_tree(X,y):
    DT = DecisionTreeClassifier(max_depth = 40)
    param_grid = {
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model, best_params = tune_hyperparameters(DT, param_grid, X, y)
    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)
    with open('output/DecisionTree.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("Cross-validated Scores fo\n")
        f.write(f"Best params for Decision Tree: {best_params}")
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()


def random_forest(X,y):
    RF = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    best_model, best_params = tune_hyperparameters(RF, param_grid, X, y)
    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)   
    with open('output/RandomForest.txt', 'w') as f:
        f.write(f"Best params for Random Forest: {best_params}")
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()



def suppor_vector_machine(X,y):
    svc = SVC(gamma='auto')
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    best_model, best_params = tune_hyperparameters(svc, param_grid, X, y)
    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)
    with open('output/SupporVMachine.txt', 'w') as f:
        f.write(f"Best params for SVC: {best_params}")
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()

def naive_bayes(X,y):
    nb = GaussianNB()
    cv_results = cross_validate(nb, X, y, cv=5, scoring=scoring)
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()


def KNN(X,y):
    knn = KNeighborsClassifier(n_neighbors=3)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    best_model, best_params = tune_hyperparameters(knn, param_grid, X, y)
    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)
    with open('output/KNearNeightboor.txt', 'w') as f:
        f.write(f"Best params for KNN: {best_params}")
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()




def xgboost(X,y):
    xgb = XGBClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    best_model, best_params = tune_hyperparameters(xgb, param_grid, X, y)
    cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)
 
    with open('output/xgboost.txt', 'w') as f:
        f.write(f"Best params for XGBoost: {best_params}")
    return cv_results['test_accuracy'].mean(),cv_results['test_precision'].mean(),cv_results['test_recall'].mean(),cv_results['test_f1'].mean()


def benchmarkbar(bench_data):
    # Transformar los datos para que se ajusten al formato adecuado para seaborn
    melted_data = bench_data.melt(id_vars='Model', var_name='Metric', value_name='Score')

    # Crear un gráfico de barras
    plt.figure(figsize=(12, 8))  # Ajustar el tamaño para mejor visualización
    barplot = sns.barplot(data=melted_data, x='Metric', y='Score', hue='Model', palette='tab10')

    # Mejoras estéticas
    plt.title('\nBenchmark\n', fontsize=26)  # Título del gráfico
    #plt.xlabel('Metric', fontsize=14)  # Etiqueta del eje X
    plt.ylabel('Score', fontsize=14)  # Etiqueta del eje Y
    plt.xlabel('Metric', fontsize=14)  # Etiqueta del eje Y
    plt.ylim(0.5, 0.95) # Ajustar los límites del eje Y para mejor enfoque

    # Ajustar la leyenda
    plt.legend(title='ML Model', fontsize=15, title_fontsize='13', loc='upper left', bbox_to_anchor=(1, 1))  # Posicionamiento de la leyenda
    plt.xticks(fontsize=20)
    plt.grid(axis='y', which='both', linestyle='--')
    plt.yticks([0.75, 0.8, 0.85, 0.9, 0.95], fontsize=12)
    
    # Ajustar el layout para evitar recortes y superposiciones
    plt.tight_layout()

    # Mostrar el gráfico
    plt.savefig('output/benchmark_bar.png', dpi=300)


def benchmark(bench_data):
    categories = list(bench_data)[1:]  # Esto excluye la primera columna que es 'Model'
    N = len(categories)

    angles = [n/float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    fig, ax = plt.subplots(figsize=(25,18), subplot_kw={'polar': True})
    plt.xticks(angles[:-1],categories, color='b',size=35)
    ax.set_rlabel_position(270)
    plt.title("\nML methods Benchmark\n",size=50)
    for i in range(bench_data.shape[0]):
        values = bench_data.loc[i].drop('Model').values.flatten().tolist()
        values += values[:1]  # Se completa el círculo
        ax.plot(angles,np.log(values), linewidth=2.5, linestyle='-', label=bench_data['Model'][i])
        ax.fill(angles,np.log(values), alpha=0.085)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if label.get_text() == 'Accuracy' or label.get_text() == 'Recall' :
            label.set_horizontalalignment('left' if angle < np.pi else 'right')
            label.set_verticalalignment('bottom' if angle < np.pi/2 or angle > 3*np.pi/2 else 'top')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(0.25, 0.25), fontsize=30)

    for line in legend.get_lines():
        line.set_linewidth(4.0) 
    ax.set_yticklabels([])
    plt.tight_layout()

    plt.savefig('output/bench.png', dpi=300)
    with open('output/benchmark_results.txt', 'w') as f:
        f.write("*******************************************************************\n")
        f.write("Benchmark Results\n")
        f.write("*******************************************************************\n\n")
        f.write(bench_data.to_string(index=False))
