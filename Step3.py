import requests
import yaml
 
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"

response_cop = requests.get(heat_pump_cop_file)
 
if response_cop.status_code == 200:
        data_cop = yaml.safe_load(response_cop.text)
        print(data_cop)
    
inputs_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/inputs.yaml"
 
response_inputs = requests.get(inputs_file)
 
if response_inputs.status_code == 200:
        data_inputs = yaml.safe_load(response_inputs.text)
        print(data_inputs)
        

