import tensorflow as tf 
import matplotlib.pyplot as plt

import numpy as np
from networkx import karate_club_graph, to_numpy_matrix
zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

print("Zackary Adjency matrix is :")
print(A_hat)

#  initialize weights randomly
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
print("first layer weight:")
print(W_1)
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))
print("second layer weight:")
print(W_2)

# def gcn_layer(A_hat, D_hat, X, W):
#     return tf.nn.relu(D_hat**-1 * A_hat * X * W)

def gcn_layer(A_hat, D_hat, X, W):
    return (D_hat**-1 * A_hat * X * W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

print("two layer GCNN output :")
print (H_2)





npH2=np.array(H_2)
x= npH2[:,0]
y= npH2[:,1]

plt.scatter(x,y,s=50)
plt.grid(True)
plt.show()
