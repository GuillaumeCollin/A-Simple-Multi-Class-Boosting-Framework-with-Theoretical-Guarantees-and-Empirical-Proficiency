import numpy as np
from keras.datasets import mnist
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import sklearn
from sklearn.svm import SVC
seed = 7
np.random.seed(seed)

## load dataset

def isolet_dataset():
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        with open('isolet_train.data', 'r') as f:
            for line in f:
                temp = line.strip('\n').split(', ')
                for i in range(0, len(temp)):
                    if i != len(temp) -1:
                        temp[i] = float(temp[i])
                    else:
                        temp[i] = int(float(temp[i]))
                X_train.append(temp[:-1])
                y_train.append(temp[-1])
      
        with open('isolet_test.data', 'r') as f:
            for line in f:
                temp = line.strip('\n').split(', ')
                for i in range(0, len(temp)):
                    if i != len(temp) -1:
                        temp[i] = float(temp[i])
                    else:
                        temp[i] = int(float(temp[i]))
                X_test.append(temp[:-1])
                y_test.append(temp[-1])

        return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    
def vowel_dataset():
        f = open('vowel-context.data', 'r')
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
                X_train.append([data[i] for i in range(1, len(data)-1)])
                y_train.append(data[-1])
            else:
                X_test.append([data[i] for i in range(1, len(data)-1)])
                y_test.append(data[-1])
        #print(X_train, y_train, X_test, y_test)
        print(X_train[0])
        return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)


## Model Neural Network

def model_nn_vowel(input_shape=784, num_classes=10):
    model = Sequential()
    model.add(Dense(4*num_classes, input_dim=input_shape, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(2*num_classes, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.00005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def model_nn_mnist(input_shape=784, num_classes=10):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_shape, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.00005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def model_nn_isolet(input_shape=784, num_classes=10):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.00005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
    
## Evaluate dataset        
def evaluate_SVM(dataset_name = 'mnist'):
    if dataset_name.find('mnist') != -1:
        print('loading mnist dataset ...')
        
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        T = X_train.shape[1]*X_train.shape[2]
        X_train = X_train.reshape(-1, T).astype('float32')
        X_test = X_test.reshape(-1, T).astype('float32')
        X_train = X_train/255.0
        X_test = X_test/255.0
        model = SVC(C=5.0, kernel='rbf', gamma=0.05)
        
    elif dataset_name.find('vowel') != -1:
        print('loading vowel ...')
        X_train, y_train, X_test, y_test = vowel_dataset()
        print('shape : ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        
        # load model
        model = SVC(C=0.001, kernel='linear')
        
    elif dataset_name.find('isolet') != -1:
        print('loading isolet ...')
        X_train, y_train, X_test, y_test = isolet_dataset()
        print('shape : ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        
        # load model
        model = SVC(C=1, kernel='linear')
         
    # Fit the model
    model.fit(X_train, y_train)
        
    # Final evaluation of the model
    scores = model.score(X_test, y_test)
    print(scores)
    print("Baseline Error: %.2f%%" % (100-scores*100))
    
    
def evaluate_nn(dataset_name = 'mnist'):
    if dataset_name.find('mnist') != -1:
        print('loading mnist dataset ...')
        
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        T = X_train.shape[1]*X_train.shape[2]
        X_train = X_train.reshape(-1, T).astype('float32')
        X_test = X_test.reshape(-1, T).astype('float32')
        X_train = X_train/255.0
        X_test = X_test/255.0
        
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classe = y_test.shape[1]
        print(num_classe)
        
        model = model_nn_mnist(input_shape=(X_train.shape[1]), num_classes=num_classe)
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=256, verbose=1)
        
    elif dataset_name.find('vowel') != -1:
        X_train, y_train, X_test, y_test = vowel_dataset()
        print('shape : ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classe = y_test.shape[1]
        print(num_classe)
        # load model
        
        model = model_nn_vowel(input_shape=(X_train.shape[1]), num_classes=num_classe)
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=4, verbose=1)
    
    elif dataset_name.find('isolet') != -1:
        X_train, y_train, X_test, y_test = isolet_dataset()
        print('shape : ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classe = y_test.shape[1]
        print(num_classe)
        # load model
        
        model = model_nn_isolet(input_shape=(X_train.shape[1]), num_classes=num_classe)    
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

#evaluate_SVM('mnist')
evaluate_nn('mnist')
    
    
## MNIST
# SVM RBF : 1.63 error
# NN : 1.78 error

## VOWEL
# SVM linear : 50.43 error
# NN : 65.80 error

## isolet
# SVM linear : 3.98 error
# NN : 3.98 1024 neurones