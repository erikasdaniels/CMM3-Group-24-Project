# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:20:13 2024

@author: erika
"""

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
print(data[['time', 'temp']])

'''
----------------------------------------------------------------------------------------------------------------
#STEP 2 BEGINS HERE
'''
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
print("---COP GRAPH---")
print(f"The gradient of the best fit line, b = {slope:.2f}" )
print(f"The y intercept of the best fit line, a = {intercept:.2f} ")

# Calculating correlation co-efficient r
cop_mean = np.mean(np_cop_values)
inverse_array_mean = np.mean(np_inverse_array)
covariance = np.sum((inverse_array - inverse_array_mean)*(np_cop_values-cop_mean))
SD_product = np.sqrt(np.sum((np_inverse_array - inverse_array_mean)**2)*np.sum((np_cop_values-cop_mean)**2))
r = covariance / SD_product 
print(f"Correlation coefficient, r= {r:.2f}")
if r == 1 or r == -1:
    print("This graph has perfect association")
elif 0.8 <= abs(r) < 1:
    print("This graph has very strong association")
elif 0.6 <= abs(r) < 0.8:
    print("This graph has strong association")
elif 0.4 <= abs(r) < 0.6:
    print("This graph has moderate association")
elif 0.2 <= abs(r) < 0.4:
    print("This graph has weak association")
else:
    print("This graph has very weak/no association")
'''
#STEP 3 STARTS HERE
--------------------------------------------------------------------------------------------------------    
'''
# Extracting values from inputs .yaml and storing them in separate variables
indoor_setpoint_temperature = data_inputs['building_properties']['indoor_setpoint_temperature_K']['value']
roof_U_value = data_inputs['building_properties']['roof_U_value']['value']
roof_area = data_inputs['building_properties']['roof_area']['value']
wall_U_value = data_inputs['building_properties']['wall_U_value']['value']
wall_area = data_inputs['building_properties']['wall_area']['value']

fixed_condenser_temperature = data_inputs['heat_pump']['fixed_condenser_temperature_K']['value']
heat_transfer_area = data_inputs['heat_pump']['heat_transfer_area']['value']
off_temperature_threshold = data_inputs['heat_pump']['off_temperature_threshold_K']['value']
on_temperature_threshold = data_inputs['heat_pump']['on_temperature_threshold_K']['value']
overall_heat_transfer_coefficient = data_inputs['heat_pump']['overall_heat_transfer_coefficient']['value']

heat_loss_coefficient = data_inputs['hot_water_tank']['heat_loss_coefficient']['value']
mass_of_water = data_inputs['hot_water_tank']['mass_of_water']['value']
specific_heat_capacity = data_inputs['hot_water_tank']['specific_heat_capacity']['value']
total_thermal_capacity = data_inputs['hot_water_tank']['total_thermal_capacity']['value']

initial_tank_temperature = data_inputs['initial_conditions']['initial_tank_temperature_K']['value']

time_step = data_inputs['simulation_parameters']['time_points']['value']
total_time = data_inputs['simulation_parameters']['total_time_seconds']['value']



delta_t_ambient =  [temp - indoor_setpoint_temperature for temp in t_ambient]



Q_load = np.abs([((wall_area * wall_U_value * t) + (roof_area * roof_U_value * t)) for t in delta_t_ambient])

plt.figure(2)
plt.plot(delta_t_ambient,Q_load )
plt.title("Plot of Q_Load against Delta T")
plt.xlabel(" Outside Temperature - Indoor Setpoint Temperature (K)")
plt.ylabel("Q_Load (W)")

'''
-------------------------------------------------------------------------------------------------------------------------
STEP 4 BEGINS HERE
'''


# Constants for the simulation
n_steps = time_step

time_step = total_time/n_steps

# Initialize variables
T_tank = initial_tank_temperature
temperature_array = np.zeros(n_steps)
time_array = np.arange(0, total_time, time_step)
heat_pump_status = [False]  

def heat_pump(t_tank, heat_pump_status):

    # Calculate the heat transfer if the pump is on
    Q_transfer = overall_heat_transfer_coefficient * heat_transfer_area * (fixed_condenser_temperature - t_tank)

    # Check the current tank temperature and update the pump status
    if t_tank < on_temperature_threshold:
        heat_pump_status = True  # Turn on the heat pump
     
        
    elif t_tank >= off_temperature_threshold:
        heat_pump_status = False  # Turn off the heat pump
       

    # Return heat transfer based on pump status
    if heat_pump_status == True:
        return Q_transfer, heat_pump_status
    else:
        return 0 , heat_pump_status  # No heat transfer when the pump is off


'''
STEP 5 BEGINS HERE 
-----------------------------------------------------------------------------------------------------------------------------------
'''
#calcualting SA from ratio of radius to height
k = 1/2 
density_water = 997 #kg/m^3
V_water = mass_of_water/density_water
tank_surface_area = 2*((V_water)**(2/3))*np.pi**(1/3)*(k**(1/3)+k**(2/3))



# Initialize the Heat transfer, load and loss lists
current_list = []
loss_list = []
load_list = []
transfer_list = []
cop_list = []
time_hours = time_array / 3600

# Define the ODE for tank temperature
def tank_temperature_ode(t, y, heat_pump_status):
    t_tank = y[0]  # Unpack the tank temperature from the state vector
    
    current_hour = int(t // 3600)
    
    # Assign the current ambient temperature from the list of hours
    t_ambient_current = t_ambient[current_hour]
    delta_t_ambient_current = t_ambient_current - indoor_setpoint_temperature
    current_list.append(t_ambient_current)
    
    cop = intercept + slope*(1/delta_t_ambient_current)
    cop_list.append(cop)
    
    # Get heat transfer from the heat pump
    Q_transfer, heat_pump_status[0] = heat_pump(t_tank, heat_pump_status[0])
    transfer_list.append(Q_transfer)
    
    # Calculate heat load and heat loss
    Q_load_current = np.abs((wall_area * wall_U_value * delta_t_ambient_current) + 
                              (roof_area * roof_U_value * delta_t_ambient_current))
    load_list.append(Q_load_current)
    
    Q_loss = float(heat_loss_coefficient * tank_surface_area * (t_tank - t_ambient_current))
    loss_list.append(Q_loss)
        
    
    # Calculate the rate of temperature change (dT/dt)
    dT_tank_dt = (Q_transfer -Q_loss - Q_load_current) / (mass_of_water * specific_heat_capacity)
    
    return [dT_tank_dt], heat_pump_status[0]  # Return both dT/dt and updated heat pump status


# Update Euler method to include status tracking
def euler_method(ode_func, y0, t_span, time_step, heat_pump_status):
    n_steps = int((t_span[1] - t_span[0]) / time_step) + 1
    t = np.linspace(t_span[0], t_span[1], n_steps)
    y = np.zeros(n_steps)
    status = []  # Initialize an empty list to track heat pump status
    y[0] = y0

    # Initialize the heat pump status for the first time step
    current_heat_pump_status = heat_pump_status[0]
    status.append(current_heat_pump_status)

    for i in range(1, n_steps):
        dYdt, current_heat_pump_status = ode_func(t[i-1], [y[i-1]], heat_pump_status)
        y[i] = y[i-1] + dYdt[0] * time_step
        status.append(current_heat_pump_status)  # Append the current heat pump status

    return t, y, status  # Return time, temperature, and heat pump status


# Initial conditions
T_tank = initial_tank_temperature
time_span = (0, total_time)

# Use Euler's method to get results
time_array, T_tank_solution, heat_pump_status_list = euler_method(
    tank_temperature_ode, T_tank, time_span, time_step, heat_pump_status)

# Find the minimum and maximum temperatures and their indices
min_temp = np.min(T_tank_solution)
max_temp = np.max(T_tank_solution)
min_index = np.argmin(T_tank_solution)
max_index = np.argmax(T_tank_solution)

# Corresponding times
min_time = time_array[min_index]
max_time = time_array[max_index]

# Print the minimum and maximum temperatures with their times in HH:MM format
min_hours = int(min_time // 3600)
min_minutes = int((min_time % 3600) // 60)
max_hours = int(max_time // 3600)
max_minutes = int((max_time % 3600) // 60)


# Plot the tank temperature
plt.figure()
plt.plot(time_hours, T_tank_solution, label='Tank Temperature (K)')
plt.xlabel('Time (hours)')
plt.ylabel('Tank Temperature (K)')
plt.title('Tank Temperature Over Time')
plt.axhline(333.15, color="red", linestyle = '--',  label="Off Temperature Threshold")
plt.axhline(313.15, color="green",linestyle = '--', label="On Temperature Threshold")
plt.scatter(min_time / 3600, min_temp, color='orange', zorder=5)
plt.scatter(max_time / 3600, max_temp, color='purple', zorder=5)
plt.xlim(0, 24)
plt.grid(True)
plt.legend(loc = 'right')

# Plot the heat pump status
plt.figure()
plt.plot(time_hours, heat_pump_status_list, label='Heat Pump Status')
plt.xlabel('Time (hours)')
plt.ylabel('Heat Pump Status')
plt.title('Heat Pump Status Over Time')
plt.grid(True)
plt.legend()
plt.show()

# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot ambient temperature on the first y-axis
ax1.plot(time_hours[:-1], current_list, color='blue', label='Ambient Temperature (K)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Ambient Temperature (K)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# Create a second y-axis for the COP values
ax2 = ax1.twinx()
ax2.plot(time_hours[:-1], cop_list, color='green', label='COP')
ax2.set_ylabel('COP', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Title and legend
plt.title('Ambient Temperature and COP Over Time')
fig.tight_layout()  # Adjust layout for better fit
plt.show()


plt.figure()

# Plotting all three datasets on the same graph
plt.plot(time_hours[:-1], loss_list, label='Heat Loss (W)', color='red')  
plt.plot(time_hours[:-1], transfer_list, label='Heat Transfer (W)', color='blue')  
plt.plot(time_hours[:-1], load_list, label='Heat Load (W)', color='green')  

# Adding labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Power (W)')
plt.title('Heat Loss, Transfer, and Load Over Time')

# Adding grid, legend, and showing the plot
plt.grid(True)
plt.legend()
plt.show()

# Total energy used by the heat pump over 24 hours
total_energy_joules = np.sum(transfer_list) * time_step  # Summing Q_transfer * time_step to get total energy in joules
total_energy_kwh = total_energy_joules / 3_600_000  # Convert joules to kWh
average_cop = np.mean(cop_list)
min_cop = np.min(cop_list)
max_cop = np.max(cop_list)


# Calculate the maximum power output of the heat pump
max_thermal_output = max(transfer_list)

# Calculate total useful thermal energy needed by the building over 24 hours (in joules)
total_useful_energy_joules = np.sum(load_list) * time_step  # Joules
total_useful_energy_kwh = total_useful_energy_joules / 3_600_000  # Convert to kWh

# Calculate system efficiency as the ratio of useful energy to electrical energy input
system_efficiency = (total_useful_energy_kwh / total_energy_kwh) * 100  # Efficiency as a percentage


print()
print("---PERFORMANCE METRICS---")
print(f'The optimised tank surface area is: {tank_surface_area:.2f} m^2 ')
print()
print(f'Minimum Temperature: {min_temp:.2f} K at {min_hours:02}:{min_minutes:02}')
print(f'Maximum Temperature: {max_temp:.2f} K at {max_hours:02}:{max_minutes:02}')
print()
print(f"Total energy usage by the heat pump over 24 hours: {total_energy_kwh:.2f} kWh")
print(f"Total useful thermal energy provided to meet heating load: {total_useful_energy_kwh:.2f} kWh")
print(f"Total system efficiency: {system_efficiency:.2f}%")
print()
print(f"Heat pump maximum thermal output: {max_thermal_output:.2f} W")
print()
print(f"Average COP over 24 hours: {average_cop:.2f}")
print(f"Maximum COP: {max_cop:.2f}, Minimum COP: {min_cop:.2f} ")

