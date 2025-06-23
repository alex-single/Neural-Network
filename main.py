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

x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), 
                                np.array(c).reshape(1, 30)]
y = np.array(y)


def sigmoid(x):
    return(1/(1 + np.exp(-x)))

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

def back_propagation(x, y, w1, w2, alpha):
    # forward pass
    z1 = x.dot(w1)            # (1×5)
    a1 = sigmoid(z1)          # (1×5)
    z2 = a1.dot(w2)           # (1×3)
    a2 = sigmoid(z2)          # (1×3)

    # output error
    d2 = a2 - y               # (1×3)

    # back-propagate into hidden layer
    d1 = d2.dot(w2.T) * (a1 * (1 - a1))  # (1×5)

    # gradients
    w2_grad = a1.T.dot(d2)   # (5×3)
    w1_grad = x.T.dot(d1)    # (30×5)

    # update weights
    w1 -= alpha * w1_grad
    w2 -= alpha * w2_grad

    return w1, w2

#dimension 30 x 1 dot 
w1 = generate_wt(30, 5)
w2 = generate_wt(5, 3)

def train(x, y , w1, w2, alpha, epoch):
    acc = []
    los = []
    for i in range(epoch):
        ls = []
        for j in range(len(x)):
            #run once
            output = f_forward(x[j], w1, w2)
            ls.append(loss(output, y[j]))
            w1, w2 = back_propagation(x[j], y[j], w1, w2, alpha)
            print("epochs:", j + 1, "======== acc:", (1-(sum(ls)/len(x)))*100) 
            acc.append((1-(sum(ls)/len(x)))*100)
            los.append(sum(ls)/len(x))
        
    return(acc, los, w1, w2)

acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 10000)

def predict(x, w1, w2):
	Out = f_forward(x, w1, w2)
	maxm = 0
	k = 0
	for i in range(len(Out[0])):
		if(maxm<Out[0][i]):
			maxm = Out[0][i]
			k = i
	if(k == 0):
		print("Image is of letter A.")
	elif(k == 1):
		print("Image is of letter B.")
	else:
		print("Image is of letter C.")
	plt.imshow(x.reshape(5, 6))
	plt.show() 
# Example: Predicting for letter 'B'	
predict(x[0], w1, w2)



import matplotlib.pyplot as plt1

# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()

# plotting Loss
plt1.plot(losss)
plt1.ylabel('Loss')
plt1.xlabel("Epochs:")
plt1.show()