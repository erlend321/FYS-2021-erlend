# task 2
import numpy as np
import matplotlib.pyplot as plt

year = []
temp = []

# read data
with open('globaltemp.csv', 'r') as data:
    for i, line in enumerate(data):
        if i < 7:
            continue
        years, temperatures = map(float, line.split(' '))
        year.append(years)
        temp.append(temperatures)

# make into arrays
year = np.array(year)
temp = np.array(temp)

plt.scatter(year, temp)
#plt.plot(year, temp, 'o') # gjør akkurat samme som den over



A = np.vstack([year, np.ones(len(year))]).T  # Creating matrix for least squares method
m, b = np.linalg.lstsq(A, temp, rcond=None)[0]  # Solve for slope (m) and intercept (b)

predicted_temp = m * year + b


"""
R2, finner ut av hvor god modellen fungerer på dataene, 
eller om det er bedre å bare bruke snittet av dataene
"""
ss_res = np.sum((temp - predicted_temp) ** 2) # residual sum of squares
ss_tot = np.sum((temp - np.mean(temp)) ** 2) # total sum of squares

R2 = 1 - (ss_res / ss_tot)

print(f"R^2 verdi er {R2:.5f}") # 0.81454


# Plot the regression line
plt.plot(year, predicted_temp, 'r')

# Add labels and title
plt.xlabel('År')
plt.ylabel('Temperatur')
plt.legend()
plt.show()






