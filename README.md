# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 20/9/2025
# Name: PRIYADHARSSHINI R



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

# Load dataset
data = pd.read_csv("IMDB Top 250 Movies (1).csv")

# Make a time series: average rating per year
X = data.groupby("year")["rating"].mean()

plt.figure(figsize=(12,6))
plt.plot(X)
plt.title("Average IMDB Rating per Year")
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title("ACF of Ratings per Year")

plt.subplot(2,1,2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title("PACF of Ratings per Year")
plt.tight_layout()
plt.show()


# Simulate ARMA(1,1)
phi_arma11 = arma11_model.params['ar.L1']
theta_arma11 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi_arma11])
ma1 = np.array([1, theta_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.plot(ARMA_1)
plt.title("Simulated ARMA(1,1) Process")
plt.show()

plot_acf(ARMA_1)
plt.title("ARMA(1,1) ACF")
plt.show()
plot_pacf(ARMA_1)
plt.title("ARMA(1,1) PACF")
plt.show()


# Simulate ARMA(2,2)
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.plot(ARMA_2)
plt.title("Simulated ARMA(2,2) Process")
plt.show()

plot_acf(ARMA_2)
plt.title("ARMA(2,2) ACF")
plt.show()
plot_pacf(ARMA_2)
plt.title("ARMA(2,2) PACF")
plt.show()
```

OUTPUT:
#### Original :
<img width="1365" height="670" alt="image" src="https://github.com/user-attachments/assets/b0b7316e-6462-4ed5-a72d-1897cee3bbbf" />


### SIMULATED ARMA(1,1) PROCESS:
<img width="989" height="556" alt="image" src="https://github.com/user-attachments/assets/b1488ed6-3d56-4d26-a30a-5757aa09cceb" />




#### Partial Autocorrelation:
<img width="1112" height="564" alt="image" src="https://github.com/user-attachments/assets/c21b3453-ceb9-4ffb-bb2d-e5336db42425" />


#### Autocorrelation:
<img width="939" height="568" alt="image" src="https://github.com/user-attachments/assets/a819c279-00c0-4b12-a0fa-da15c2d5806c" />




### SIMULATED ARMA(2,2) PROCESS:
<img width="926" height="584" alt="image" src="https://github.com/user-attachments/assets/226dd5b7-cd75-4f97-a275-e78bf492c514" />


#### Partial Autocorrelation:
<img width="931" height="566" alt="image" src="https://github.com/user-attachments/assets/5262ee13-cf87-4ce0-8c52-e2e45b70bf9c" />




#### Autocorrelation:
<img width="1014" height="566" alt="image" src="https://github.com/user-attachments/assets/62b715eb-9d3d-4a90-9c42-87f56756632e" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
