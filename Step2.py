
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

# Load YAML file into a DataFrame
with open('C:/Users/erika/Documents/Engineering/Computational Methods and Modeling Material/Project/Downloads/heat_pump_cop_synthetic_full.yaml', 'r') as f:
    data = safe_load(f)  # Load the YAML file as a dictionary

# Check the structure of the loaded data
print("Loaded Data:", data)

# Normalize the YAML data into a DataFrame
df = pd.json_normalize(data)

# Print the DataFrame to inspect its structure
print("DataFrame Structure:\n", df)

plt.plot('COP_noisy','outdoor_temp_C')
