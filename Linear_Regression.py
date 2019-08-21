import numpy as np

# creating our training set
class Dataset:
    @staticmethod
    def create_dataset_x():
        a = np.array([[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]])
        return a
    @staticmethod
    def create_dataset_y():
        b = np.array([[2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0]])
        return b


    
# algorithm ignition
class LinearRegression:
    ys=np.ndarray(shape=(1,10),dtype=float)
    xs=np.ndarray(shape=(1,10),dtype=float)
    
    def __init__(self,  t0,  t1,  alpha):
        self.t0=t0
        self.t1=t1
        self.alpha=alpha
        dataset = Dataset()
        self.xs=dataset.create_dataset_x()
        self.ys=dataset.create_dataset_y()
        self.theta = np.array([[t1]])
        
    def calculate(self,Theta,Xs):
        th= Xs.dot(Theta)
        temp=0
        for i in range(0,10):
            temp=th[i,0]+temp
        print('temp is: ',temp)    
        return temp
    
    def ignit(self):
        h=float(self.calculate(Theta=self.theta,Xs=self.xs.transpose()))
        alp=float(self.alpha*(1/10))
        for i in range(0,10):
            temp0=float(self.t0-(alp*(h-self.ys[0,i])))
            temp1=float(self.t1-(alp*(h-self.ys[0,i])*self.xs[0,i]))
            self.t0=temp0
            self.t1=temp1
        print('theta values:')
        print('theta0: ',self.t0,'theta1: ',self.t1)  
        if(self.t0==0 and self.t1==2):
            pass
        else:
            self.ignit()    

  

lr=LinearRegression(0,1,1)
lr.ignit()

