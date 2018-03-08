from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from localised_similarities import LocalisedSimilaritiesClassifier
from sklearn.utils.estimator_checks import check_estimator
from REBEL import REBEL
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

import pandas as pd

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

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

    return x_train.values, x_test.values, y_train.values, y_test.values

def synthetic_dataset(nb_points, dim, nb_classes):
    x = np.zeros((nb_classes*nb_points,dim))
    y = np.zeros(nb_classes*nb_points)
    #r = np.linspace(0.5, 0.7, nb_classes)
    r= 0.5
    teta = np.linspace(0,12,nb_points)
    abscisse = np.linspace(-1, 1, nb_points)
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

    return x_train, x_test, y_train, y_test


def __main__():

#    x_train, x_test, y_train, y_test=get_data('glass/glass.data')

    ##Boosting
    #ada=AdaBoostClassifier(base_estimator=LocalisedSimilaritiesClassifier(),algorithm='SAMME')
    #ada.fit(x_train,y_train)
#    print(x_train)
 #   boosting = REBEL(max_iteration=100)
  #  boosting.fit(x_train, y_train,x_test,y_test)
    #print(boosting.score(x_test,y_test))

    x_train, x_test, y_train, y_test = synthetic_dataset(100,2,2)

    print('y_train')
    print(y_train)
    #graph(x_train[:,0],x_train[:,1],y_train,'test')

    # x_train, x_test, y_train, y_test = get_data('pendigits/pendigits.tra')

    ##Boosting
    # ada=AdaBoostClassifier(base_estimator=LocalisedSimilaritiesClassifier(),algorithm='SAMME')
    # ada.fit(x_train,y_train)
    boosting = REBEL(max_iteration=4)
    boosting.fit(x_train, y_train, x_test, y_test)
    # print(boosting.score(x_test,y_test))

__main__()
