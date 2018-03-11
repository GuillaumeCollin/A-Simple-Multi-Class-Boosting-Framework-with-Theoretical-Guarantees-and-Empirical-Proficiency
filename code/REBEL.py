from localised_similarities import LocalisedSimilaritiesClassifier
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

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
        self.liste_xi = []
        self.liste_xj = []
        self.resultats = []

    def compute_H(self,x, H):
        a_t = np.zeros((x.shape[0],self.n_classes))
        for element in range(self.current_iteration):
            a_t[:] = self.array_a[element]
            for xi in range(x.shape[0]):
                H[:, xi] += self.compute_weak_learner(self.array_f[element],x)[xi] * self.array_a[element]
        return H


    def compute_weak_learner(self, vecteur_carac,x):
        tau = vecteur_carac[1]
        xi = self.liste_xi[vecteur_carac[2]]
        xj = self.liste_xj[vecteur_carac[3]]
        if vecteur_carac[0] == 0:
            f = np.ones(x.shape[0])

        if vecteur_carac[0] == 1:
            x_norme = []
            for element in x:
                x_norme.append(np.linalg.norm(xi - element) ** 2)
            f = (tau - np.array(x_norme)) / (tau + np.array(x_norme))


        if vecteur_carac[0] == 2:
            f = []
            d = 1 / 2 * (xi - xj)
            m = 1 / 2 * (xi + xj)
            for element in x:
                scalaire = np.dot(d, (element - m))
                x_norme = np.linalg.norm(element - m) ** 4
                f.append(scalaire / (4 * np.linalg.norm(d) ** 4 + np.array(x_norme)))
            f = np.array(f)

        ans = np.ones(x.shape[0])
        ans[f > 0] = 1  # Peut être plutot mettre une classe observée, mais laquelle ?
        ans[f < 0] = -1
        return ans

    def fit(self,x, y, x_test, y_test):
        self._params_init(x,y)
        for iteration in range(self.max_iteration):
            print('--------------------Nouvelle iteration----------------------')
            week_clf = self.clf.fit(x,y,self.H)

            params = week_clf.get_params()
            self.array_a[iteration] = params['a']
            dict = params['localised_similarity']
            self.array_f[iteration][0] = dict['type']
            self.array_f[iteration][1] = dict['tau']
            self.liste_xi.append(dict['xi'])
            self.liste_xj.append(dict['xj'])
            #self.H = dict['H']

            if self.final_loss is None:
                self.final_loss = params['loss']
            else :
                loss = params['loss']
                if loss < self.final_loss:
                    self.final_loss = loss
                else:
                    print('la loss ne semble plus descendre')
                    break
            if iteration%30 == 0:
                X, Y = self.decision_border(50)
                graph(X[:, 0], X[:, 1], Y, 'glass_border'+str(iteration))

            print('loss dans REBEL')
            print(self.final_loss)
            print('Current iteration')
            print(self.current_iteration)
            self.current_iteration += 1
            print('Score train')
            score_train = self.score(x, y)
            print(score_train)
            print('Score test')
            score_test = self.predict_test(x_test, y_test)
            print(score_test) # il faut recalculer H pour tous les points

            self.resultats.append([self.current_iteration-2, self.final_loss, 1- score_train,1 -score_test])

        print("Training is over")

    def get_resultats(self):
        return self.resultats

    def get_prediction(self):
        return self.test_predicions

    def predict_test(self,x,y):
        H = np.zeros((self.n_classes, x.shape[0]))
        predictions = []

        for iteration in range(self.current_iteration-1):
            weak_learner=self.compute_weak_learner([self.array_f[iteration][0],10**(-2),iteration,iteration],x)
            a = self.array_a[iteration]
            for xi in range(x.shape[0]):
                H[:, xi] += weak_learner[xi] * a
        for index in range(x.shape[0]):
            predictions.append(np.argmax(H[:,index]))
        self.test_predicions = predictions
        foo = predictions - y
        score = foo[foo == 0].shape[0] / x.shape[0]
        return score

    def predict_border(self, x):
        H = np.zeros((self.n_classes, x.shape[0]))
        predictions = []
        for iteration in range(self.current_iteration - 1):
            weak_learner = self.compute_weak_learner([self.array_f[iteration][0], 10 ** (-2), iteration, iteration], x)
            a = self.array_a[iteration]
            for xi in range(x.shape[0]):
                H[:, xi] += weak_learner[xi] * a
        for index in range(x.shape[0]):
            predictions.append(np.argmax(H[:, index]))
        border_predicions = predictions
        return border_predicions

    def predict(self,x):
        predictions = []
        for index in range(x.shape[0]):
            predictions.append(np.argmax(self.H[:,index]))

        return predictions

    def score(self,x_test,y_test):
        predictions = self.predict(x_test)
        foo = predictions - y_test
        score = foo[foo == 0].shape[0]/x_test.shape[0]
        return score

    def decision_border(self,nb_point):
        X = self.decision_function(nb_point)
        Y = self.predict_border(X)
        return X,Y

    def decision_function(self,nb_point):
        X = []
        abscisse = np.linspace(-1,1,nb_point)
        ordonnee = np.linspace(-1, 1, nb_point)
        for element_x in abscisse:
            for element_y in ordonnee:
                X.append([element_x,element_y])
        print(np.array(X))
        return np.array(X)

def main():

    x_train = np.array([[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9]])
    x_test = np.arange(0, 10).reshape(-1,1)
    y_train = np.ones(10)
    y_train[0:5] = 0
    y_train[9] = 2
    print(y_train)
    boosting = REBEL(max_iteration=100000)
    boosting.fit(x_train, y_train,x_train,y_train)
    prediction = boosting.predict(x_test)
    print('y_train, prediction finale')
    print(y_train)
    print(prediction)

# main()

#Two point localised ne fonctionne probablement pas