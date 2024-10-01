import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #only used for splitting



#loading data
data = pd.read_csv('data_problem2.csv')
data = data.transpose().reset_index()
data.columns = ["value", "Label"]
# print(data["value"])

#convert the values to floats
data["value"] = data["value"].astype(float)

#check type == float (we dont want string)
# print(data["value"].to_numpy().shape)
# print(type(data["value"].to_numpy()[0]))



#dividing the data into their own arrays
C0_data = data[data["Label"] == 0]["value"].to_numpy()  #class 0
C1_data = data[data["Label"] == 1]["value"].to_numpy()  #class 1


"""
Plotting different histograms
One for all the data and then two different sets representing c0 and c1
"""

#all the datapoints
plt.figure(figsize=(8, 6))
plt.hist(data["value"].to_numpy(), bins = 100, edgecolor='black')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)



#overlapping the histograms for C0 and C1
plt.figure(figsize=(8, 6))
plt.hist(C0_data, bins=100, color='blue', alpha=0.6, label='Class C0', edgecolor='black')
plt.hist(C1_data, bins=100, color='yellow', alpha=0.6, label='Class C1', edgecolor='black')
plt.title('Overlapping Histogram of C0 and C1')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)



"""
Task 2b, finding the MSE estimations of the parameters
"""

N_C0 = len(C0_data)
N_C1 = len(C1_data)

alfa = 2

def beta_hat():
    return (1 / (N_C0 * alfa)) * np.sum(C0_data)

print(f"Estimate for beta for C0: {beta_hat():.4f}\n")

def mean_hat():
    return (1 / N_C1) * np.sum(C1_data)

print(f"Estimated mean for C1: {mean_hat():.4f}\n")

def square_sigma_hat():
    return (1 / N_C1) * np.sum((C1_data - mean_hat())**2)

print(f"Estimated sigma**2 for C1: {square_sigma_hat():.4f}\n")



"""
2c, splitting data into training and test 80/20
"""

x = np.array(data['value'])
y = np.array(data['Label'])

#shuffle and split, using the same method as I did in the last assignment
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, stratify = y, random_state = 10)

print(f"Train set:  {len(x_train)}")
print(f"Test set:  {len(x_test)}\n")




def gamma(n):

    factorial = 1
    for i in range(1, n):
        factorial *= i

    return factorial


def gamma_distrubution(beta, x):
    return (1 / ((beta**alfa) * gamma(alfa)))  * (x**(alfa-1))  * np.exp(-x / beta)


def gaussian_distribution(sigma, x, mean):
    return (1/ (sigma * np.sqrt(2*np.pi))) * np.exp((-1/2)*((x-mean)/sigma)**2)



def bayes_classifier(x, beta_hat, mean_hat, sigma_hat):
    #likelihood for c0
    P_C0 = gamma_distrubution(beta_hat, x)

    #likelihoood for c1
    P_C1 = gaussian_distribution(sigma_hat, x, mean_hat)

    #return the 'correct' class for the one with higher likelihood
    if P_C1 > P_C0:
        return 1
    else:
        return 0
    
#we use our bayes classifier to sort the values
predictions = []

for i in x_test:
    prediction = bayes_classifier(i, beta_hat(), mean_hat(), np.sqrt(square_sigma_hat()))
    predictions.append(prediction)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy on test set:  {accuracy*100:.3f}%\n")


#storing the mis,- and the correctly classified results
misclassified = x_test[np.array(predictions) != y_test]
correctly_classified = x_test[np.array(predictions) == y_test]


print(f"Misclassified values: {len(misclassified)}\n")
print(f"Correctly classified values: {len(correctly_classified)}\n")


"""
2d, plotting of miscalculated and correctly calculated values
"""

plt.figure(figsize=(8, 6))

#histogram for correctly classified values
plt.hist(correctly_classified, bins=50, color='green', alpha=0.6, label='Correctly Classified', edgecolor='black')

#histogram for misclassified values
plt.hist(misclassified, bins=50, color='red', alpha=0.6, label='Misclassified', edgecolor='black')

plt.title('Histogram of Correctly Classified vs Misclassified Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()









