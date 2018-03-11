from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from localised_similarities import LocalisedSimilaritiesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from REBEL import REBEL
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.datasets import make_gaussian_quantiles

import pandas as pd


def vowel_dataset():
    f = open('data/VOWEL/vowel-context.data', 'r')
    datas = []
    cpt = 0
    for line in f:
        temp = line.split()
        for i in range(14):
            if i == 0 or i == 1 or i == 2 or i == 13:
                temp[i] = int(temp[i])
            else:
                temp[i] = float(temp[i])
        if temp[0] == 0:
            cpt += 1
        datas.append(temp)

    f.close()
    print(datas[0])
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for data in datas:
        if data[0] == 0:
            X_train.append([data[i] for i in range(1, len(data) - 1)])
            y_train.append(data[-1])
        else:
            X_test.append([data[i] for i in range(1, len(data) - 1)])
            y_test.append(data[-1])
    # print(X_train, y_train, X_test, y_test)
    print(X_train[0])
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

def graph(x,y,classe,title):

    # Scatter plot
    trace = go.Scatter(

        x=x,
        y=y,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            size=25,
            color=classe,
            colorscale='Portland',
            showscale=True
        ),
        text=x
    )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        title=title,
        hovermode='closest',
        #     xaxis= dict(
        #         title= 'Pop',
        #         ticklen= 5,
        #         zeroline= False,
        #         gridwidth= 2,
        #     ),
        yaxis=dict(
            title='Classes',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='output/'+title+'.html')


def get_data(name):

    data = pd.read_csv('data/'+name,header=None)
   # train = modify_target(train)
    X = data.drop([0, 10], axis=1)
    Y = data[10]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=43)

    return x_train.values, x_test.values, y_train.values, y_test.values

def synthetic_dataset(nb_points, dim, nb_classes):
    x = np.zeros((nb_classes*nb_points,dim))
    y = np.zeros(nb_classes*nb_points)
    #r = np.linspace(0.5, 0.7, nb_classes)
    r= 0.5
    teta = np.linspace(0,12,nb_points)

    for classe in range(nb_classes):
        #sans bruit
        abscisse = 1/10 * teta * np .cos(teta+ 2*np.pi*classe/nb_classes)
        ordonnes = 1/10 * teta * np .sin(teta+ 2*np.pi*classe/nb_classes)
        #avec bruit
        x[classe*nb_points:(classe+1)*nb_points,0] = abscisse
        x[classe*nb_points:(classe+1)*nb_points,1] = ordonnes
        y[classe*nb_points:(classe+1)*nb_points] = classe
        # t = np.linspace(classe * 4, (classe + 1) * 4, nb_points) + 0.2 * randn(1, ns);
        #r*cos(x) est l'équation d'une spirale

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)

    return x_train, x_test, y_train, y_test

def synthetic_dataset_2():
    x = decision_function(20)
    y = np.zeros(x.shape[0])

    for element in range(x.shape[0]):
        if x[element][1] <= int(x[element][0]):
           y[element] = 0
        else:
           y[element]= 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)

    return x_train, x_test, y_train, y_test

def synthetic_dataset_1(nb_points, dim, nb_classes):

    nb_points_1 = 50
    nb_points_2 = 100
    nb_points_3 = 150
    x = np.zeros((nb_classes * nb_points, dim))
    y = np.zeros(nb_classes * nb_points)
    # r = np.linspace(0.5, 0.7, nb_classes)
    r = 0.5

    teta_1 = np.linspace(0, 12, nb_points_1)

    abscisse_1 = 1 / 10 * teta_1 * np.cos(teta_1 + 2*np.pi*1/nb_classes )
    ordonnes_1 = 1 / 10 * teta_1 * np.sin(teta_1 + 2*np.pi*1/nb_classes )

    x[0:nb_points_1, 0] = abscisse_1
    x[0:nb_points_1, 1] = ordonnes_1
    y[0:nb_points_1 ] = 0


    teta_2 = np.linspace(0, 12, nb_points_2)

    abscisse_2 = 1 / 10 * teta_2 * np.cos(teta_2 + 2*np.pi*2/nb_classes)
    ordonnes_2 = 1 / 10 * teta_2 * np.sin(teta_2 + 2*np.pi*2/nb_classes)
    # avec bruit
    x[nb_points_1:nb_points_2+nb_points_1, 0] = abscisse_2
    x[nb_points_1:nb_points_2+nb_points_1, 1] = ordonnes_2
    y[nb_points_1:nb_points_2+nb_points_1] = 1

    teta_3 = np.linspace(0, 12, nb_points_3)

    abscisse_3 = 1 / 10 * teta_3 * np.cos(teta_3 )
    ordonnes_3 = 1 / 10 * teta_3 * np.sin(teta_3 )
    # avec bruit
    x[nb_points_1+nb_points_2:, 0] = abscisse_3
    x[nb_points_1+nb_points_2:, 1] = ordonnes_3
    y[nb_points_1+nb_points_2:] = 2


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)

    return x_train, x_test, y_train, y_test



def mislabelled(y,nb):
    classes = np.unique(y)
    nb_classe = classes.shape[0]
    indices_a_changer = np.random.randint(0, y.shape[0], nb)
    for indices in indices_a_changer:
        new_classe = np.random.randint(0, nb_classe)
        while new_classe == y[indices]:
            new_classe = np.random.randint(0, nb_classe)
        y[indices] = new_classe
    return y

def normalisation(x):
    new_x = np.zeros((x.shape[0],x.shape[1]))
    for column in range (x.shape[1]):
        new_x[:,column] = x[:,column]/np.sum(x[:,column])
    return new_x * 100

def decision_function(nb_point):
    X = []
    abscisse = np.linspace(-1,1,nb_point)
    ordonnee = np.linspace(-1,1, nb_point)
    for element_x in abscisse:
        for element_y in ordonnee:
            X.append([element_x,element_y])
    print(np.array(X))
    return np.array(X)

def detection_overfitting():
    def true_fun(X):
        return np.cos(1.5 * np.pi * X) + 3
    X = decision_function(30)
    y = np.zeros(X.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    x_train = X
    y_train = y
    y_test = y
    bruit = np.random.randn(x_train.shape[0]) * 0.5
    index = []

    for indice in range(x_train.shape[0]):
        if x_train[indice,1] > true_fun(x_train[indice,0]) + bruit[indice]:
            y_train[indice] = 1
        else:
            y_train[indice] = 0
        if x_train[indice,1] > 4.5 or x_train[indice,1] < 1.5:
            index.append(indice)

    x_train = np.delete(x_train, index, axis = 0)
    y_train = np.delete(y_train, index)
    y_test = np.delete(y_test, index)

    graph(x_train[:, 0], x_train[:, 1], y_train, 'synthetic_train_bruit')

    for indice in range(x_train.shape[0]):
        if x_train[indice, 1] > true_fun(x_train[indice, 0]):
            y_test[indice] = 1
        else:
            y_test[indice] = 0
        if x_train[indice, 1] > 4.5 or x_train[indice, 1] < 1.5:
            index.append(indice)

    graph(x_train[:, 0], x_train[:, 1], y_test, 'synthetic_test_bruit')

    return x_train, x_train, y_train, y_test

def __main__():

  #  x_train, x_test, y_train, y_test=get_data('glass/glass.data')

 #   x_train = normalisation(x_train)

  #  x_test = normalisation(x_test)

    ##Boosting
    #ada=AdaBoostClassifier(base_estimator=LocalisedSimilaritiesClassifier(),algorithm='SAMME')
    #ada.fit(x_train,y_train)
#    print(x_train)
 #   boosting = REBEL(max_iteration=100)
  #  boosting.fit(x_train, y_train,x_test,y_test)
    #print(boosting.score(x_test,y_test))

  #  x_train, x_test, y_train, y_test = synthetic_dataset(100,2,3)

    #y_train = y_train
    #print(x_train)
    #print('y_train')
    #print(y_train)
    #

   # x_train, x_test, y_train, y_test = get_data('glass/glass.data')

    x_train, x_test, y_train, y_test = detection_overfitting()

    ##Boosting
    # ada=AdaBoostClassifier(base_estimator=LocalisedSimilaritiesClassifier(),algorithm='SAMME')
    # ada.fit(x_train,y_train)
    graph(x_train[:,0],x_train[:,1],y_train,'synthetic_train')
    boosting = REBEL(max_iteration = 301)
    boosting.fit(x_train, y_train, x_test, y_test)
    y_predict_test = boosting.get_prediction()
    print(y_predict_test)
    graph(x_test[:,0],x_test[:,1],y_predict_test,'synthetic_test')


    resultats = np.array(boosting.get_resultats())
    plt.plot(resultats[:,[0]],resultats[:,[1]],label='Loss')
    plt.plot(resultats[:,[0]],resultats[:,[2]],label='erreur_train')
    plt.plot(resultats[:,[0]],resultats[:,[3]],label='erreur_test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
   # boosting.fit(x_train, y_train, x_test, y_test)

# print(boosting.score(x_test,y_test))

def main2():
    #x_train, x_test, y_train, y_test = synthetic_dataset(150,2,3)
    x_train, x_test, y_train, y_test = get_data('pendigits/pendigits.tra')
    graph(x_train[:, 0], x_train[:, 1], y_train, 'synthetic_test')
    knn = KNeighborsClassifier(3)
    knn.fit(x_train, y_train)
    X = decision_function(50)
    Z = knn.score(x_test,y_test)
    print(Z)
    #graph(X[:, 0], X[:, 1], Z, 'synthetic_test_1')

#__main__()

def main3():
    X1, y1 = make_gaussian_quantiles(cov=2.,
                                     n_samples=200, n_features=2,
                                     n_classes=2, random_state=1)
    X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    graph(x_train[:, 0], x_train[:, 1], y_train, 'gaussian_train')
    boosting = REBEL(max_iteration=301)
    boosting.fit(x_train, y_train, x_test, y_test)
    y_predict_test = boosting.get_prediction()
    print(y_predict_test)
    graph(x_test[:, 0], x_test[:, 1], y_predict_test, 'gaussian_test')

    resultats = np.array(boosting.get_resultats())
    plt.plot(resultats[:, [0]], resultats[:, [1]], label='Loss')
    plt.plot(resultats[:, [0]], resultats[:, [2]], label='erreur_train')
    plt.plot(resultats[:, [0]], resultats[:, [3]], label='erreur_test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def main4():
    X_train, y_train, X_test, y_test = vowel_dataset()
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    score = ada.score(X_test,y_test)
    print(score)

main2()