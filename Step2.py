import yaml

# Load the YAML file
with open(r"C:\Users\erika\Documents\Engineering\Computational Methods and Modeling Material\Project\Downloads\heat_pump_cop_synthetic_full.yaml", 'r') as file:
    data = yaml.safe_load(file)

# Extract COP_noisy and outdoor_temp_C into arrays
cop_values = [entry['COP_noisy'] for entry in data['heat_pump_cop_data']]
outdoor_temp_values = [entry['outdoor_temp_C'] for entry in data['heat_pump_cop_data']]

# Now cop_values and outdoor_temp_values are Python lists (arrays)
print("COP values:", cop_values)
print("Outdoor temperatures:", outdoor_temp_values)
