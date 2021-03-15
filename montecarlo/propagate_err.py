from scipy.optimize import curve_fit, OptimizeWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings
from inspect import signature

warnings.simplefilter("error", OptimizeWarning)

class MonteCarloError:
    def __init__(self,x,y,popt,func,xerr=0,yerr=0,iterations=1000):
        sig = signature(func)
        self.funcArgs = list(dict(sig.parameters).keys())[1:]
        self.iterations = iterations
        self.x = x
        self.y = y
        self.xerr = xerr
        if type(xerr) == int:
            self.xerr = np.zeros(len(x))
        self.yerr = yerr
        if type(yerr) == int:
            self.yerr = np.zeros(len(x))
        self.func = func
        self.popt = popt

    def genDataSet(self):
        Ys = []
        for m in range(len(self.x)):
            Ys.append(np.random.normal(loc=self.y[m],scale=self.yerr[m],size=self.iterations))
        Ys = np.array(Ys).T
        self.Ys = Ys
        Xs = []
        for m in range(len(self.x)):
            Xs.append(np.random.normal(loc=self.x[m],scale=self.xerr[m],size=self.iterations))
        Xs = np.array(Xs).T
        self.Xs = Xs
        for k in range(len(Ys)):
            plt.plot(Xs[k],Ys[k],marker="x",linestyle="",color="k")
        plt.show()
        return Xs,Ys
    
    def optimize(self):
        popts = []
        for k in range(self.iterations):
            try:
                popt, pcov = curve_fit(self.func,self.Xs[k],self.Ys[k],p0=self.popt)#,sigma=o_E_offsets)
                popts.append(popt)
            except:
                pass
        popts = np.array(popts)
        self.popt_std = np.std(popts,axis=0)
        
    def run(self):
        self.genDataSet()
        self.optimize()
        # print("average popt",self.popt_avg)
        # print("stdev popt",self.popt_std)
        print("params:")
        for k in range(len(self.popt)):
            print(self.funcArgs[k]+" = ",self.popt[k],"+-",self.popt_std[k])
        return self.popt_std




if __name__ == "__main__":
    def func1(x,a,b): 
        return a*x + b

    x = np.array([ k for k in range(20) ])
    y = func1(x,2,3) + np.random.normal(1,1,20)
    o_x = np.ones(20)*0.3
    o_y = np.ones(20)*0.5
    
    popt,pcov = curve_fit(func1,x,y)
    
    plt.plot(x,y)
    plt.show()
    mce = MonteCarloError(x,y,popt,func1,xerr=o_x,yerr=o_y,iterations=10000)
    err = mce.run()
