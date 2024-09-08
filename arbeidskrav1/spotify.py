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



reduced_data = data[(data['genre'] == 'Pop') | (data['genre'] == 'Classical')].copy()

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)   



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



def predict(A, weight):
    return sigmoid_funk(np.dot(A, weight))


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

        # regner ut cost og lagrer det i cost_list
        cost = compute_cost(A, y, weight)
        cost_list.append(cost)

        # printer cost ved gitt antall epochs
        if e % 10 == 0: 
            print(f"Epoch nr. {e} Cost: {cost:.5f}\n")

    return weight, cost_list

def accuracy(A, y, weight):
    guess = predict(A, weight) >= .5 
    accuracy = np.mean(guess == y) * 100
    return accuracy


# adding bias
A_train_bias = np.c_[np.ones((x_train.shape[0], 1)), x_train]
weight = np.zeros(A_train_bias.shape[1]) # start weights at 0


lr = .0001   # 0.0001
epoch = 120

weight, cost_list = SGD(A_train_bias, y_train, weight, lr, epoch)

# plotter training error

plt.plot(range(epoch), cost_list)
plt.xlabel('Epochs')
plt.ylabel('Cost (Error)')
plt.title('How much error do we have')
plt.show()

train_accuracy = accuracy(A_train_bias, y_train, weight)
print(f"Training accuracy {train_accuracy:.4f}%")

# legger til bias til test set
# np.c_ is to create stacked columnwise arrays see https://images.datacamp.com/image/upload/v1676302459/Marketing/Blog/Numpy_Cheat_Sheet.pdf
A_test_bias = np.c_[np.ones((x_test.shape[0], 1)), x_test]

# accuracy on test set
test_accuracy = accuracy(A_test_bias, y_test, weight)
print(f"Test accuracy:  {test_accuracy:.3f}% right")


"""
Confusion matrix
"""

def confusion_matrix(y_true, y_pred):
    # initialiserer verdiene til 0
    TP = TN = FP = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    return [[TN, FP], [FN, TP]]

# predictions based on test set
A_test_bias = np.c_[np.ones((x_test.shape[0], 1)), x_test]
y_test_pred = predict(A_test_bias, weight) >= .5
cm = confusion_matrix(y_test, y_test_pred)

print(f"Confusion matrix:  {cm}")
