'''
THIS IS A COPY OF STEP1.PY, WITHOUT IT, STEP3.PY WILL NOT RUN INDIVIDUALLY. THIS WON'T BE A PROBLEM IN THE
MASTER FILE, SO THIS BIT CAN BE REMOVED  :)
'''
from datetime import datetime
from meteostat import Point, Hourly

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
    
'''

-----------------------------------------------------------------------

'''
import requests
import yaml
import matplotlib.pyplot as plt
 
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"

response_cop = requests.get(heat_pump_cop_file)
 
if response_cop.status_code == 200:
        data_cop = yaml.safe_load(response_cop.text)
    
inputs_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/inputs.yaml"
 
response_inputs = requests.get(inputs_file)
 
if response_inputs.status_code == 200:
        data_inputs = yaml.safe_load(response_inputs.text)
        
# Extracting values and storing them in separate variables
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

time_points = data_inputs['simulation_parameters']['time_points']['value']
total_time_seconds = data_inputs['simulation_parameters']['total_time_seconds']['value']

delta_t_ambient = [indoor_setpoint_temperature - temp for temp in t_ambient]


q_load = [((wall_area * wall_U_value * t) + (roof_area * roof_U_value * t)) for t in delta_t_ambient]


plt.plot(t_ambient,q_load )

