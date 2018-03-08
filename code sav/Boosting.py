from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from localised_similarities import LocalisedSimilaritiesClassifier
from sklearn.utils.estimator_checks import check_estimator

import pandas as pd

def get_data(name):

    data = pd.read_csv('data/'+name,header=None)
   # train = modify_target(train)
    X = data.drop([0, 10], axis=1)
    Y = data[10]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

    return x_train.values, x_test.values, y_train.values, y_test.values

def __main__():

    ok = check_estimator(LocalisedSimilaritiesClassifier)

    print(ok)

    x_train, x_test, y_train, y_test=get_data('glass/glass.data')

    ##Boosting
    ada=AdaBoostClassifier(base_estimator=LocalisedSimilaritiesClassifier(),algorithm='SAMME')
    ada.fit(x_train,y_train)
__main__()
