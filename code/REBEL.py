from localised_similarities import LocalisedSimilaritiesClassifier
import numpy as np

class REBEL(object):

    def __init__(self,max_iteration=1000, clf=LocalisedSimilaritiesClassifier()):
        self.max_iteration = max_iteration
        self.clf = clf

    def _params_init(self,x,y):
        self.x_train = x
        self.y_train = y
        self.current_iteration = 1
        self.n_classes = np.unique(y).shape[0]
        self.array_a = np.zeros((self.max_iteration, self.n_classes))
        self.array_f = np.zeros((self.max_iteration, 4))
        self.H = np.zeros((self.n_classes, x.shape[0]))
        self.final_loss = None

    def compute_H(self,x, H):
        a_t = np.zeros((x.shape[0],self.n_classes))
        for element in range(self.current_iteration):
            a_t[:] = self.array_a[element]
            for xi in range(x.shape[0]):
                H[:, xi] += self.compute_weak_learner(self.array_f[element],x)[xi] * self.array_a[element]
        return H

    def compute_weak_learner(self, vecteur_carac,x):
        tau = vecteur_carac[1]
        xi = vecteur_carac[2]
        xj = vecteur_carac[3]
        if vecteur_carac[0] == 0:
            return np.ones(x.shape[0])

        if vecteur_carac[0] == 1:
            x_norme = []
            for element in x:
                x_norme.append(np.linalg.norm(xi - element) ** 2)
            f = (tau - np.array(x_norme)) / (tau + np.array(x_norme))
            return f

        if vecteur_carac[0] == 2:
            f = []
            d = 1 / 2 * (xi - xj)
            m = 1 / 2 * (xi + xj)
            for element in x:
                scalaire = np.dot(d, (element - m))
                x_norme = np.linalg.norm(element - m) ** 4
                f.append(scalaire / (4 * np.linalg.norm(d) ** 4 + np.array(x_norme)))
            return np.array(f)

        raise

    def fit(self,x, y,x_test,y_test):
        self._params_init(x,y)
        for iteration in range(self.max_iteration):
            print('--------------------Nouvelle iteration----------------------')
            week_clf = self.clf.fit(x,y,self.H)

            params = week_clf.get_params()
            self.array_a[iteration] = params['a']
            dict = params['localised_similarity']
            self.array_f[iteration][0] = dict['type']
            self.array_f[iteration][1] = dict['tau']
            #self.array_f[iteration][2] = dict['xi']
            #self.array_f[iteration][3] = dict['xj']
            #self.H = dict['H']

            if self.final_loss is None:
                self.final_loss = params['loss']
            else :
                loss = params['loss']
                if loss < self.final_loss:
                    self.final_loss = loss
                else :
                    break
            print('Current iteration')
            print(self.current_iteration)
            self.current_iteration += 1
            print('Score train')
            print(self.score(x, y))
            print('Score test')
            print(self.score(x_test, y_test))

        print("Training is over")

    def predict(self,x):
        H = self.H
        predictions = []
        for index in range(x.shape[0]):
            predictions.append(np.argmax(H[:,index]))

        return predictions

    def score(self,x_test,y_test):
        predictions = self.predict(x_test)
        foo = predictions - y_test
        score = foo[foo == 0].shape[0]/x_test.shape[0]
        return score

    def decision_function(self):
        return 'Pas codée'

def main():

    x_train = np.array([[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9]])
    x_test = np.arange(0, 10).reshape(-1,1)
    y_train = np.ones(10)
    y_train[4:6] = 0
    y_train[9] = 0
    y_train[0] = 2
    print(y_train)
    boosting = REBEL(max_iteration=100000)
    boosting.fit(x_train, y_train)
    prediction = boosting.predict(x_test)
    print('y_train, prediction finale')
    print(y_train)
    print(prediction)

# main()

#Two point localised ne fonctionne probablement pas