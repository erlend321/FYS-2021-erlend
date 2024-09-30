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
plt.figure(figsize=(3, 2))
plt.hist(data["value"].to_numpy(), bins = 50, edgecolor='black')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()


#plotter c0
plt.figure(figsize=(3, 2))
plt.hist(C0_data, bins = 30, edgecolor='black')
plt.title('Histogram of C0 values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()

#plotter c1
plt.figure(figsize=(3, 2))
plt.hist(C1_data, bins = 30, edgecolor='black')
plt.title('Histogram of C1 values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
#plt.show()


"""
Task 2b, finding the MSE estimations of the parameters
"""

N_C0 = len(C0_data)
N_C1 = len(C1_data)

alfa = 2

def beta_hat():
    return (1 / (N_C0 * alfa)) * np.sum(C0_data)

print(f"Our estimate for beta for C0: {beta_hat():.4f}\n")

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
print(f"Test set:  {len(x_test)}")


