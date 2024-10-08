import yaml
import matplotlib.pyplot as plt
import numpy as np
import requests 
 
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"

response_cop = requests.get(heat_pump_cop_file)
 
if response_cop.status_code == 200:
        data_cop = yaml.safe_load(response_cop.text)
    
        
# Extract COP_noisy and outdoor_temp_C into arrays
cop_values = [entry['COP_noisy'] for entry in data_cop['heat_pump_cop_data']]
outdoor_temp_values = [entry['outdoor_temp_C'] for entry in data_cop['heat_pump_cop_data']]

# Now cop_values and outdoor_temp_values are Python lists (arrays)
#print("COP values:", cop_values)
#print("Outdoor temperatures:", outdoor_temp_values)

plt.plot(outdoor_temp_values,cop_values)
 
plt.cla()

plt.scatter(outdoor_temp_values,cop_values)


delta_t = [60 - value for value in outdoor_temp_values]
inverse_array = [1 / x for x in delta_t]

#print(delta_t)

#print(inverse_array)

plt.cla()

np_delta_t = np.array(delta_t)
np_inverse_array = np.array(inverse_array)
np_cop_values = np.array(cop_values)



slope, intercept = np.polyfit(inverse_array , cop_values, 1)

line_of_best_fit = slope * np_inverse_array + intercept

plt.scatter(inverse_array , cop_values)

plt.plot(inverse_array,line_of_best_fit)

print("b =" ,slope)
print("a =" , intercept)

print(slope)
print(intercept)

print(slope)
print(intercept)
