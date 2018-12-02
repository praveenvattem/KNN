import pandas as pd
import numpy as np

#generate random numpy array
mean = np.array([5, 1])
cov = [[1000, 300], [200, 100]]  # diagonal covariance
import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, size =(1024,5000)).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


#create train set
X=x.copy()
X_train=X[924:1024]
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)


# training SOM
from minisom import MiniSom
som =MiniSom(x=10,y=10,input_len=1024,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X_train)
som.train_random(data= X_train,num_iteration=500)

#plot the graph
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers =['o','s']
colors=['r','g']
for a,b in enumerate(X_train):
    w=som.winner(b)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[a]],
         markeredgecolor = colors[y[a]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()