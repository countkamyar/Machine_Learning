import numpy as np

# creating our training set
class Dataset:
    @staticmethod
    def create_dataset():
        a = np.array([[1.0, 5.0], [2.0, 7.0], [3.0, 9.0],[4.0,11.0],[5.0,13.0]])
        return a

# hypothesis function
class Hypothesis:
    Theta=np.ndarray(shape=(1,1))
    Xs=np.ndarray(shape=(5,1),dtype=float)
    def __init__(self, Theta, Xs):
        self.Theta = Theta
        self.Xs = Xs

    def calculate(self):
        th= self.Xs.dot(self.Theta)
        temp=0
        for i in range(0,5):
            temp=th[i,0]+temp
        return temp

# algorithm ignition
class LinearRegression:
    ys=np.ndarray(shape=(1,5),dtype=float)
    xs=np.ndarray(shape=(1,5),dtype=float)
    A=np.ndarray(shape=(5,2),dtype=float)
    def __init__(self,  t0,  t1,  alpha):
        self.t0=t0
        self.t1=t1
        self.alpha=alpha
        dataset = Dataset()
        self.A=dataset.create_dataset()
        for i in range(0,5):
           self.xs[0,i]=self.A[i,0]
           self.ys[0,i]=self.A[i,1]
        print('xs',self.xs)
        self.theta = np.array([[t1]])
        
    def ignit(self):
        hx=Hypothesis(Theta=self.theta,Xs=self.xs.transpose())
        h=float(hx.calculate())
        al=float(self.alpha*(self.A.shape[0]))
        for i in range(0,5):
            temp0=float(self.t0-(al*(h-self.ys[0,i])))
            temp1=float(self.t1-(al*(h-self.ys[0,i])*self.xs[0,i]))
            self.t0=temp0
            self.t1=temp1
        print('theta values:')
        print('theta0: ',self.t0,'theta1: ',self.t1)  
        if(self.t0==self.t1):
            pass
        else:
            self.ignit()    

  

lr=LinearRegression(0,1,1)
print(lr.ignit())

