

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

#STEP 3 STARTS HERE    

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

time_points = data_inputs['simulation_parameters']['time_points']['value']
total_time_seconds = data_inputs['simulation_parameters']['total_time_seconds']['value']


delta_t_ambient = [indoor_setpoint_temperature - temp for temp in t_ambient]


q_load = [((wall_area * wall_U_value * t) + (roof_area * roof_U_value * t)) for t in delta_t_ambient]

plt.figure(2)
plt.plot(t_ambient,q_load )
plt.title("Plot of Q_Load against Ambient Temperature")
plt.xlabel("Outside Ambient Temperature (K)")
plt.ylabel("Q_Load (W)")


