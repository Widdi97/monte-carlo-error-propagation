import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from montecarlo import propagate_err

# function which is fitted
def func1(x,a,b): 
    return a*x + b

# generate a rng data set
x = np.array([ k for k in range(20) ])
y = func1(x,2,3) + np.random.normal(1,1,20)
o_x = np.ones(20)*0.3 # error of x
o_y = np.ones(20)*0.5 # error of y

# fit
popt,pcov = curve_fit(func1,x,y)

# plot
plt.plot(x,y,linestyle="",marker="x")
plt.plot(x,func1(x,*popt))
plt.show()

# monte carlo error propagation
mce = propagate_err.MonteCarloError(x,y,popt,func1,xerr=o_x,yerr=o_y,iterations=100)
err = mce.run()