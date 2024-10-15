from datetime import datetime
from meteostat import Point, Hourly
import numpy as np
import yaml
import matplotlib.pyplot as plt
import requests


# Define the location (Edinburgh: 55.9533° N, 3.1883° W)
location = Point(55.9533, -3.1883)

# Set the time period for data retrieval
start = datetime(2023, 1, 1, 0) # Start time
end = datetime(2023, 1, 2, 0) # End time (24-hour period)

# Fetch hourly temperature data
data = Hourly(location,start,end)
data = data.fetch()
data = data.reset_index()
t_ambient = []
hours = data["time"].dt.hour
for i in range(0,25):
    t = data["temp"].iloc[i] + 273 
    t_ambient.append(t)
    

print("")
print("Here we can see the data that we've extracted from the meteostat library in the time period 1st Jan to 2nd Jan 2023:")
print("")
print(data)

#STEP 2 BEGINS HERE

# Load the YAML file directly from github
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"

response_cop = requests.get(heat_pump_cop_file)

if response_cop.status_code == 200:
        data_cop = yaml.safe_load(response_cop.text)

inputs_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/inputs.yaml"

response_inputs = requests.get(inputs_file)

if response_inputs.status_code == 200:
        data_inputs = yaml.safe_load(response_inputs.text)


# Extract COP_noisy and outdoor_temp_C into arrays
cop_values = [entry['COP_noisy'] for entry in data_cop['heat_pump_cop_data']]
outdoor_temp_values = [entry['outdoor_temp_C'] for entry in data_cop['heat_pump_cop_data']]


delta_t = [60 - value for value in outdoor_temp_values]
inverse_array = [1 / x for x in delta_t]

np_delta_t = np.array(delta_t)
np_inverse_array = np.array(inverse_array)
np_cop_values = np.array(cop_values)

slope, intercept = np.polyfit(inverse_array , cop_values, 1)

line_of_best_fit = slope * np_inverse_array + intercept

plt.figure(1)
plt.scatter(inverse_array , cop_values, marker = "x")

plt.plot(inverse_array,line_of_best_fit, "r")

plt.xlabel("1/Delta_T")
plt.ylabel("Coefficient of Performance")
plt.title("Plot of COP vs 1/Delta_T")

print("")
print("The gradient of the best fit line, b =" ,slope)
print("The y intercept of the best fit line, a =" , intercept)

# Calculating correlation co-efficient r
cop_mean = np.mean(np_cop_values)
inverse_array_mean = np.mean(np_inverse_array)
covariance = np.sum((inverse_array - inverse_array_mean)*(np_cop_values-cop_mean))
SD_product = np.sqrt(np.sum((np_inverse_array - inverse_array_mean)**2)*np.sum((np_cop_values-cop_mean)**2))
r = covariance / SD_product 
print("Correlation coefficient, r=", r)
if r == 1 or r == -1:
    print("Perfect association")
elif 0.8 <= abs(r) < 1:
    print("Very strong association")
elif 0.6 <= abs(r) < 0.8:
    print("Strong association")
elif 0.4 <= abs(r) < 0.6:
    print("Moderate association")
elif 0.2 <= abs(r) < 0.4:
    print("Weak association")
else:
    print("Very weak/no association")
