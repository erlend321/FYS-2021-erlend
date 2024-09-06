import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # used only for splitting the dataset
import matplotlib.pyplot as plt

"""
Data prosessing part
"""

data = pd.read_csv('SpotifyFeatures.csv')


num_samples = data.shape[0]  #rader
num_features = data.shape[1]  # kolonner

"""
Hvor mange samples har vi
"""

print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")



reduced_data = data[(data['genre'] == 'Pop') | (data['genre'] == 'Classical')]

# Pop = 1 Classical = 0
reduced_data['label'] = reduced_data['genre'].apply(lambda x: 1 if x == 'Pop' else 0)


pop_samples = len(reduced_data[reduced_data['label'] == 1])
classical_samples = len(reduced_data[reduced_data['label'] == 0])

"""
Hvor mange pop og klassiske samples
"""

print(f"Number of Pop samples: {pop_samples}")
print(f"Number of Classical samples: {classical_samples}")




"""
From the reduced dataset, make 2 numpy arrays. The first array will be the matrix with songs along the
rows and songs features ("liveness" and "loudness") as columns. This will be the input of our machine
learning method. The second array will the vector with the songs genre (labels or target we want to
learn). Create a training and test set by splitting the dataset. 
"""


x = np.array(reduced_data[['liveness', 'loudness']])
y = np.array(reduced_data['label'])


# shuffle og split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)   #(x, y, test_size = 0.2, shuffle, stratify = y)   # shuffle



print(f"Training set size: {x_train.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")


"""
The machine learning part
We have to use a SGD method and could choose our own logistic discrimination
classifier. I chose to use the sigmoid function for the logistic regression,
because it is spesifically used in cases to map the output value between 1 and 0. 

"""

def sigmoid_funk(x):
    return 1/(1 + (np.e)**(-x))

# test
test = sigmoid_funk(3)
print(f"Tester om sigmoid_funk fungerer: {test:.2f} = 0,95")


def predict(A, weight):
    return sigmoid_funk(np.dot(A, weight))

# 
def compute_cost(A, y, weight):
    N = len(y)
    p = predict(A, weight)
    # binary cross-entropy. It measures the error between predicted values
    # and the labels. Will penalize incorrect predictions
    # https://www.google.com/url?sa=i&url=https%3A%2F%2Farize.com%2Fblog-course%2Fbinary-cross-entropy-log-loss%2F&psig=AOvVaw2FMNHjpl2F2UHLX1QFtssN&ust=1725702297180000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKDlu6CErogDFQAAAAAdAAAAABAE
    cost = -(1/N) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return cost

def SGD(A, y, weight, lr, epoch):
    N = len(y) # amount of samles
    cost_list = [] # saving the cost after every epoch

    for e in range(epoch):
        cost = 0 # reset
        for n in range(N):
            xn = A[n, :] 
            yn = y[n]

            p = predict(xn, weight)
            stigning = (p - yn) * xn

            # oppdaterer vekten
            weight -= lr * stigning

            





