##Fichier du weak learner

import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LocalisedSimilaritiesClassifier(object):

    def __init__(self):
        self.only_one_localized_similarities = False
        self.clf_name = None
        self.a = None

    def get_params(self, deep=True):
        params = {
            "a" : self.a,
            "localised_similarity" : {'type': self.clf_name,
                                      'tau': self.tau,
                                      'xi': self.xi,
                                      'xj': self.xj,
                                      'H': self.H},
            "loss" : self.final_loss
        }
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _params_init(self,x_train, y_train , H ):
        self.x_train = x_train
        self.y_train = y_train
        self.len_data = x_train.shape[0]
        #Probablement faux, n_target deoit être égal à 2
        self.n_target = np.unique(y_train).shape[0]
        self.Y = np.ones((self.n_target, self.len_data))  # Transformer Y pour que soit égal à 1 - 2deltayn
        self.dict = dict() #Garder la correspondance des classes
        for indice in range(self.n_target):
            self.Y[indice, y_train == np.unique(y_train)[indice]] = -1
            self.dict[indice] = int(np.unique(y_train)[indice])
        if H is not None:
            self.H = H
        else:
            self.H = np.zeros((self.n_target, self.len_data)) #Résultat du boosting à l'étape n-1
        self.W = 1/2*np.exp(self.Y * self.H)
        self.U = np.zeros((self.n_target, self.len_data))
        self.a = np.ones((self.n_target,1))
        self.clf = None
        self.y_predict = np.empty(self.len_data)
        self.B = np.ones(self.len_data)

        ##Paramètres des localised similarities
        self.xi = None
        self.xj = None
        self.tau = 10**(-2)
        self.final_loss = None


    def apply(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def fit(self, x, y, H=None):

        x, y = check_X_y(x, y)
        self.classes_ = unique_labels(y)

        ###Step 0
        self._params_init(x, y, H)
        print('L0')
        print(1 / self.len_data * np.sum(self.W))

        ###First step
        self.clf = self.f_homogeneous
        first_loss = self.optimal_loss()
        self.final_loss = first_loss

        ###Second Step
        U = self.compute_U()
        eigen_values, eigen_vectors =np.linalg.eig(U)

        v1 = eigen_vectors[:,np.argmax(eigen_values)]
        for i in range(self.len_data):
            intermediate_vector = np.zeros(self.len_data)
            intermediate_vector[i] = 1
            compute = np.dot(v1, intermediate_vector)
            if compute > 0:
                self.B[i] = 1
            else:
                self.B[i] = -1

        ###Third Step
        liste_loss = []
        self.clf = self.f_one_point
        for element in x:
            self.xi = element
            # 2 calculs de f_one_point en 1 xi pb à cause de la loss
            liste_loss.append(self.optimal_loss_binarised())
        erreurs_localised = min(liste_loss)

        #Si plusieurs bonnes localised
        indices = np.where(np.array(liste_loss) == erreurs_localised)[0]
        loss_provisoire = []
        for element in indices:
            self.xi = self.x_train[element]
            loss_provisoire.append(self.optimal_loss())
        indice_loss =loss_provisoire.index(min(loss_provisoire))
        indice_i = indices[indice_loss]
        self.xi = self.x_train[indice_i]
        second_loss = self.optimal_loss()

        #On garde la meilleure réduciton de loss

        if second_loss < self.final_loss:
            self.final_loss = second_loss
            self.clf = self.f_one_point
        else:
            self.clf = self.f_homogeneous

        xj_save = 0
        x_train_modif = self.x_train
        B_modif = self.B

        print('B')
        print(self.B)

        print('On arrive aux Two-points')
        while x_train_modif[B_modif == -B_modif[indice_i]].shape[0] > 1:
            ###Fourth Step

            x_likely_similar = x_train_modif[B_modif == -B_modif[indice_i]] - self.xi
            x_likely_similar_trans = []
            if len(x_likely_similar) > 0:
                for element in x_likely_similar:
                    x_likely_similar_trans.append(np.linalg.norm(element)**2)
                indice_j = np.argmin(x_likely_similar_trans)
                self.xj = x_train_modif[B_modif == -B_modif[indice_i]][indice_j]
            else:
                break

            ###Fifth step
            save_clf = self.clf
            self.clf = self.f_two_points
            third_loss = self.optimal_loss()

            ###Sith step
            liste_index=[]
            f_ij = self.f_two_points(x_train_modif)
            for index in range(x_train_modif.shape[0]):
                if f_ij[index] <= f_ij[indice_j]/2:
                    liste_index.append(index)

            if third_loss < self.final_loss:
                self.final_loss = third_loss
                self.clf = self.f_two_points
                xj_save = self.xj
            else:
                self.clf = save_clf
                self.xj = xj_save

            x_train_modif = np.delete(x_train_modif, liste_index)
            B_modif = np.delete(B_modif, liste_index)
            #mise à jour de l'indice_i (si décalage)
            indice_i = np.where(x_train_modif == self.xi)

        #Pour set le nom
        self.H = self.compute_H(self.x_train,self.H)
        self.clf(np.array([0]))
        self.a = self.optimal_vector()

        return self

    def base_loss(self):
        L = 1/self.len_data * np.sum(self.W*np.exp((self.compute_localised(self.x_train)*self.Y) * self.a))
        return L

    def base_loss_init(self):
        L = 1 / (2 * self.len_data) * np.sum(np.exp(self.Y*self.compute_localised(self.x_train)))
        return L

    def optimal_loss(self):

        #According to the older H

        intermediate_vector = np.zeros((self.n_target, self.x_train.shape[0]))
        intermediate_vector[self.compute_localised(self.x_train) * self.Y < 0] = np.ones(intermediate_vector[self.compute_localised(self.x_train) * self.Y < 0].shape[0])
        sf_T = 1/self.len_data * np.sum(intermediate_vector * self.W,axis=1)

        intermediate_vector = np.zeros((self.n_target, self.x_train.shape[0]))
        intermediate_vector[self.compute_localised(self.x_train) * self.Y > 0] = np.ones(intermediate_vector[self.compute_localised(self.x_train) * self.Y > 0].shape[0])
        sf_F = 1/self.len_data * np.sum(intermediate_vector * self.W,axis=1)

        loss = 2 * np.sum(np.sqrt(sf_T*sf_F))
        print(self.predict(self.x_train))
        print(loss)
        return loss

    def optimal_loss_1(self):

        #According to the older H
        new_H = np.copy(self.H)
        new_H = self.compute_H(self.x_train,new_H)
        new_W = 1/2*np.exp(self.Y * new_H)
        loss = 1/self.len_data * np.sum(new_W)

        print('loss')
        print(loss)
        print(self.optimal_loss_1())
        print('new one')
        return loss

    def optimal_loss_binarised(self):

        #According to the older H

        foo = self.B - self.compute_localised(self.x_train)
        error_1 = foo[foo == 0].shape[0]
        foo = self.B + self.compute_localised(self.x_train)
        error_2 = foo[foo == 0].shape[0]

        loss = min(error_1,error_2)
        return loss

    def optimal_vector(self):

        intermediate_vector = np.zeros((self.n_target, self.x_train.shape[0]))
        intermediate_vector[self.compute_localised(self.x_train) * self.Y < 0] = np.ones(intermediate_vector[self.compute_localised(self.x_train) * self.Y < 0].shape[0])
        sf_T = 1/self.len_data * np.sum(intermediate_vector * self.W,axis=1)

        intermediate_vector = np.zeros((self.n_target, self.x_train.shape[0]))
        intermediate_vector[self.compute_localised(self.x_train) * self.Y > 0] = np.ones(intermediate_vector[self.compute_localised(self.x_train) * self.Y > 0].shape[0])
        sf_F = 1/self.len_data * np.sum(intermediate_vector * self.W,axis=1)

        #Pour eviter les erreurs de loss on transforme les valeurs nulle en 1 pour annuler ln
        sf_T[sf_T==0] = 0.1
        sf_F[sf_F==0] = 0.1

        a = 1 / 2 * (np.log(sf_T) - np.log(sf_F))  # Permet d'attribuer les poids
        print(a)
        return a

    def compute_U(self):
        u = np.zeros((self.n_target,self.len_data))
        denom = np.sqrt(np.sum(self.W,axis=1))
        for element in range(self.len_data):
            u[:, element] = self.W[:,element]*self.Y[:,element]/denom
        return np.dot(u.T,u)

    def f_homogeneous(self,x):
        self.clf_name = 0
        return np.ones(x.shape[0])

    def f_one_point(self,x):
        self.clf_name = 1
        x_norme = []
        for element in x:
            x_norme.append(np.linalg.norm(self.xi-element) ** 2)
        f = (self.tau - np.array(x_norme))/(self.tau + np.array(x_norme))
        return f

    def f_two_points(self,x):
        self.clf_name = 2
        f = []
        d = 1/2*(self.xi-self.xj)
        m = 1/2*(self.xi+self.xj)
        for element in x:
            scalaire = np.dot(d,(element-m))
            x_norme = np.linalg.norm(element-m)**4
            f.append(scalaire/(4*np.linalg.norm(d)**4+np.array(x_norme)))
        return np.array(f)
       # return np.ones(x.shape[0])

    def compute_localised(self, x):
        # Check is fit had been called
        check_is_fitted(self, ['x_train', 'y_train'])
        #Test input
        x = check_array(x)
        ans = np.ones(x.shape[0])
        ans[self.clf(x) > 0] = 1 #Peut être plutot mettre une classe observée, mais laquelle ?
        ans[self.clf(x) < 0] = -1
        return ans

    def predict(self,x):
        #According to new H

        H = np.copy(self.H)
        H = self.compute_H(x,H)
        predictions = []
        for index in range(x.shape[0]):
            predictions.append(self.dict[np.argmax(H[:,index])])
        return predictions

    def compute_H(self,x,H):

    #    if np.all(self.H == 1):
    #        a_t = self.optimal_vector()
    #        for xi in range(x.shape[0]):
    #            H[:, xi] = self.clf(x)[xi] * a_t
    #    else:
        a_t = self.optimal_vector()

        compute = self.compute_localised(x)
        for xi in range(x.shape[0]):
            H[:, xi] += compute[xi] * a_t
        return H


    def score(self, x, y):
        predictions = self.predict(x)
        score = y-predictions
        return score[score == 0].shape


def main(name):

    x_train = np.arange(0, 10).reshape(-1,1)
    x_test = np.arange(0, 10).reshape(-1,1)
    y_train = np.ones(10)
    y_train[4:5] = 0
    clf = LocalisedSimilaritiesClassifier()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    print(y_train)
    print(prediction)
   # graph(x_train,y_train,name)
   # graph(x_test,prediction,name + '_prediction')

#main('test')
#PAs d'utilisation de W avec H et de a Dans adaboost
#Faire un dictionnaire des classes
#classes_ and n_classes_ attributes needed
#Attention base loss dans homogeneous stump
