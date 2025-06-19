import numpy as np
import matplotlib.pyplot as plt

a =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]
# B
b =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]
# C
c =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]

# Creating labels
y =[[1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]]

plt.imshow(np.array(b).reshape(5,6))
plt.show()


def sigmoid(x):
    return(1/1 + np.exp(-x))

def f_forward(x, w1, w2):
    inputlayer = x.dot(w1)
    outfirst = sigmoid(inputlayer)
    secondlayer = outfirst.dot(w2)
    outsecond = sigmoid(secondlayer)
    return outsecond

def generate_wt(x,y):
    li = []
    for i in range(x * y):
        li.append(np.random.randn())

    weight_matrix = np.array(li).reshape(x,y)

    return weight_matrix

def loss(out, y):
    mean_square_error = np.square(out - y)
    mean_square_error = np.sum(mean_square_error)/len(y)
    return mean_square_error

