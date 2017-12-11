import math
import numpy as np
from numpy.linalg import inv

#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = parameters
        self.train_x = np.empty((3,3,))
        self.train_x[0,:] = [1.0,60.0,155.0] 
        self.train_x[1,:] = [1.0,64.0,135.0] 
        self.train_x[2,:] = [1.0,73.0,170.0] 
        for i in range(1,3):
            self.train_x[:,i] = ( self.train_x[:,i] - self.train_x[:,i].mean() ) / (self.train_x[:,i].std(ddof=1))
        print 'normalized x'
        print self.train_x
        self.train_y = np.empty((1,3))
        self.train_y = [0.0,1.0,1.0]
    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    #******************************************************
    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        return ll
    #******************************************************
    def gradients(self):
        gradients = np.zeros((1, 3))
        #print gradients
        x = self.train_x
        y = self.train_y
        ##################### Please Fill Missing Lines Here #####################
        for j in range(0,len(self.parameters)):
            for i in range(0,len(self.parameters)):
                p = exp(np.dot(self.parameters,x[i,:]))
                p = p/(1+p)
                gradients[0,j] += x[i][j]*(y[i]-p)
        print 'Gradients:'
        print gradients
        return gradients
    #******************************************************
    def iterate(self,gradients,hessian):
        ##################### Please Fill Missing Lines Here #####################
        self.parameters = self.parameters - (np.dot(inv(hessian),gradients.transpose())).transpose()
        #print self.parameters
        return self.parameters
    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        x = self.train_x
        y = self.train_y
        hessian = np.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
        for i in range(0,n): 
            for j in range(0,n):
                for m in range(0,n):
                    p = exp(np.dot(self.parameters,x[m]))
                    #print p
                    p = p/(1+p)
                    hessian[i][j] -= x[m][j]*x[m][i] * p * (1-p)
                    
        print 'Hessian:'
        print hessian      
        return hessian
#-------------------------------------------------------------------



        
if __name__ == '__main__':
    parameters = []
    ##################### Please Fill Missing Lines Here #####################
    ## initialize parameters
    parameters = np.empty((1,3,))
    parameters = [0.25,0.25,0.25]
    l = logistic(parameters)
    hessian = l.hessian()
    gradients = l.gradients()
    parameters = l.iterate(gradients,hessian)
    parameters2 = np.empty((1,3))
    parameters2 = [parameters[0][0],parameters[0][1],parameters[0][2]]
    print 'iter parameter:'
    print parameters2
    l = logistic(parameters2)
    hessian = l.hessian()
    gradients = l.gradients()
    parameters2 = l.iterate(gradients,hessian)
    parameters3 = np.empty((1,3))
    parameters3 = [parameters2[0][0],parameters2[0][1],parameters2[0][2]]
    print 'iter2 parameter:'
    print (parameters3)
