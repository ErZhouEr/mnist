import random
import numpy as np

class Perceptron(object):

    def __init__(self,input_size):
        self.weights=np.array([0.0 for _ in range(input_size)])
        self.bias=0.0

    def train_params(self,X,Y,eta,N):
        for i in range(N):
            for x,y in zip(X,Y):
                cost=y-self.active_func(x)
                print(cost)
                self.weights=self.weights+eta*cost*np.array(x)
                self.bias=self.bias+eta*cost


    def active_func(self,x):
        z=sum(self.weights*np.array(x))+self.bias
        if z>0:
            return 1
        else:
            return 0

    def predict(self,X):
        return self.active_func(X)

if __name__=='__main__':
    perceptron=Perceptron(2)
    X=[[1,1],[1,0],[0,1],[0,0]]
    Y=[1,0,0,0]
    perceptron.train_params(X=X,Y=Y,eta=0.1,N=10)
    print('*************************')
    print(perceptron.predict([1,1]))
    print(perceptron.predict([1, 0]))
    print(perceptron.predict([0, 1]))
    print(perceptron.predict([0, 0]))
    print(perceptron.weights,perceptron.bias)
