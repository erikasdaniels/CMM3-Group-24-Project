import requests
import yaml
 
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

