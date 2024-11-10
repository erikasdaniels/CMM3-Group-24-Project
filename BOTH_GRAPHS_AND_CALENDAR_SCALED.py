
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import requests
import yaml
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import matplotlib.pyplot as plt


# URLs to the YAML files
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"
inputs_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/inputs.yaml"
building_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/BUILDING_VALUES.yaml"

# Fetch the YAML data
response_cop = requests.get(heat_pump_cop_file)
response_inputs = requests.get(inputs_file)
response_building = requests.get(building_file)

if response_cop.status_code == 200:
    data_cop = yaml.safe_load(response_cop.text)

if response_inputs.status_code == 200:
    data_inputs = yaml.safe_load(response_inputs.text)
    
    if response_inputs.status_code == 200:
        building_data = yaml.safe_load(response_building.text)


class GUI:
    
    def __init__(self):
        # Configure the main window
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Task B - User Interface - Group 24")

        # Target resolution
        target_width, target_height = 2560, 1200
        
        # Get screen resolution
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate scaling factors for width and height
        width_scale = screen_width / target_width
        height_scale = screen_height / target_height
        self.scale = min(width_scale, height_scale)  # Use the smaller scaling factor

        # Set window size according to screen resolution
        self.root.geometry(f"{int(target_width * self.scale)}x{int(target_height * self.scale)}")

        # Title label with scaling applied to font size
        self.label = ctk.CTkLabel(self.root, text='Heat Pump Model - Group 24', 
                                  font=("Arial", int(48 * self.scale), 'underline'))
        self.label.pack(pady=(int(20 * self.scale), int(20 * self.scale)))

        # Frame for input boxes on the left
        input_frame = ctk.CTkFrame(self.root)
        input_frame.pack(side="left", padx=(int(40 * self.scale), int(10 * self.scale)), pady=int(20 * self.scale))

        # Heating adjustment label
        adjusting_label = ctk.CTkLabel(input_frame, text='Adjusting The Heating', 
                                       font=("Arial", int(18 * self.scale), 'underline'))
        adjusting_label.grid(row=0, column=0, columnspan=2, pady=(int(10 * self.scale), int(5 * self.scale)), sticky="W")

        # Slider for indoor temperature set point with scaled dimensions
        self.indoor_temp_label = ctk.CTkLabel(input_frame, text='Indoor Temperature Set Point (°C):')
        self.indoor_temp_label.grid(row=1, column=0, sticky="W", pady=(int(10 * self.scale), int(5 * self.scale)))

        self.indoor_temp_slider = ctk.CTkSlider(input_frame, from_=10, to=30, number_of_steps=20)
        self.indoor_temp_slider.set(23)
        self.indoor_temp_slider.grid(row=2, column=0, pady=(0, int(5 * self.scale)), sticky="W")

        self.current_value = self.indoor_temp_slider.get()
        self.current_value_label = ctk.CTkLabel(input_frame, text=f"{self.current_value:.1f} °C", text_color="#0078D7")
        self.current_value_label.grid(row=3, column=0, pady=(0, int(20 * self.scale)), sticky="W")

        self.indoor_temp_slider.configure(command=self.update_slider_value)
        self.update_slider_value(self.indoor_temp_slider.get())

        # Setting Simulation Date label
        model_parameters_label = ctk.CTkLabel(input_frame, text='Setting the Simulation Date', 
                                              font=("Arial", int(18 * self.scale), 'underline'))
        model_parameters_label.grid(row=4, column=0, columnspan=2, pady=(0, int(10 * self.scale)), sticky="W")
        
        # Date frame and input fields
        date_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        date_frame.grid(row=5, column=0, columnspan=2, sticky="W", pady=(0, int(10 * self.scale)))

        self.day_entry = ctk.CTkEntry(date_frame, width=int(50 * self.scale), placeholder_text="DD")
        self.day_entry.grid(row=0, column=0, padx=(0, int(5 * self.scale)), pady=(0, int(10 * self.scale)))

        self.month_entry = ctk.CTkEntry(date_frame, width=int(50 * self.scale), placeholder_text="MM")
        self.month_entry.grid(row=0, column=1, padx=(int(5 * self.scale), int(5 * self.scale)), pady=(0, int(10 * self.scale)))

        self.year_entry = ctk.CTkEntry(date_frame, width=int(70 * self.scale), placeholder_text="YYYY")
        self.year_entry.grid(row=0, column=2, padx=(int(5 * self.scale), 0), pady=(0, int(10 * self.scale)))

        # Model Parameters label
        model_parameters_label = ctk.CTkLabel(input_frame, text='Model Parameters', 
                                              font=("Arial", int(18 * self.scale), 'underline'))
        model_parameters_label.grid(row=8, column=0, columnspan=2, pady=(0, int(10 * self.scale)), sticky="W")

        # Input labels and entry fields with scaled size
        self.inputs = [
            ("Roof U Value", "W/m²K"),
            ("Roof Area", "m²"),
            ("Wall U Value", "W/m²K"),
            ("Wall Area", "m²"),
            ("Mass of Water in Hot Water Tank", "kg"),
            ("Initial Tank Temperature", "K"),
            ("Heat Pump On", "K"),
            ("Heat Pump Off", "K"),
            ("Heat Loss Coefficient", "W/K"),
            ("Heat Transfer Coefficient", "W/K"),
        ]
        
        self.entries = {}
        for i, (label_text, unit) in enumerate(self.inputs):
            label = ctk.CTkLabel(input_frame, text=f"{label_text} ({unit}):")
            label.grid(row=i + 9, column=0, sticky="W", pady=(int(5 * self.scale), int(5 * self.scale)))
            
            entry = ctk.CTkEntry(input_frame, width=int(100 * self.scale))
            entry.grid(row=i + 9, column=1, pady=(int(5 * self.scale), int(5 * self.scale)), padx=(int(10 * self.scale), 0))
            entry.insert(0, "")
            self.entries[label_text] = entry
        
        self.populate_initial_values()

        # Adjust the columns in input_frame to accommodate centering
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Plot and Heat Transfer buttons arranged in a centered 2x2 grid
        self.execute_button = ctk.CTkButton(input_frame, text="Plot Tank Temperature", command=self.execute, width=int(150 * self.scale))
        self.execute_button.grid(row=len(self.inputs) + 9, column=0, pady=(int(20 * self.scale), int(10 * self.scale)), padx=(0, int(10 * self.scale)), sticky="EW")
        
        self.heat_button = ctk.CTkButton(input_frame, text="Clear Figure", command=self.clear_figure, width=int(150 * self.scale))
        self.heat_button.grid(row=len(self.inputs) + 9, column=1, pady=(int(20 * self.scale), int(10 * self.scale)), padx=(int(10 * self.scale), 0), sticky="EW")
        
        self.clear_button = ctk.CTkButton(input_frame, text="Plot Heat Transfer", command=self.heat, width=int(150 * self.scale))
        self.clear_button.grid(row=len(self.inputs) + 10, column=0, pady=(int(10 * self.scale)), padx=(0, int(10 * self.scale)), sticky="EW")
        
        self.reset_button = ctk.CTkButton(input_frame, text="Reset Inputs", command=self.reset_inputs, width=int(150 * self.scale))
        self.reset_button.grid(row=len(self.inputs) + 10, column=1, pady=(int(10 * self.scale)), padx=(int(10 * self.scale), 0), sticky="EW")

        # Figure frame with adjusted size
        figure_frame = ctk.CTkFrame(self.root)
        figure_frame.pack(side="right", fill="both", expand=True, padx=int(10 * self.scale), pady=int(20 * self.scale))

        self.fig = Figure(figsize=(10.67 * self.scale, 8 * self.scale), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.root.mainloop()
        
    def populate_initial_values(self):
            """Populate input fields with values from inputs.yaml."""
            values_from_yaml = {
                "Roof U Value": data_inputs["building_properties"]["roof_U_value"]["value"],
                "Roof Area": data_inputs["building_properties"]["roof_area"]["value"],
                "Wall U Value": data_inputs["building_properties"]["wall_U_value"]["value"],
                "Wall Area": data_inputs["building_properties"]["wall_area"]["value"],
                "Mass of Water in Hot Water Tank": data_inputs["hot_water_tank"]["mass_of_water"]["value"],
                "Initial Tank Temperature": data_inputs["initial_conditions"]["initial_tank_temperature_K"]["value"],
                "Heat Pump On": data_inputs["heat_pump"]["on_temperature_threshold_K"]["value"],
                "Heat Pump Off": data_inputs["heat_pump"]["off_temperature_threshold_K"]["value"],
                "Heat Loss Coefficient": data_inputs["hot_water_tank"]["heat_loss_coefficient"]["value"],
                "Heat Transfer Coefficient": data_inputs["heat_pump"]["overall_heat_transfer_coefficient"]["value"],
                }
    
            # Populate each entry field
            for label, entry in self.entries.items():
                if label in values_from_yaml:
                    entry.insert(0, values_from_yaml[label])
        
    def reset_inputs(self):
        """Reset input fields to initial values from inputs.yaml."""
        for entry in self.entries.values():
            entry.delete(0, 'end')  # Clear existing text
        self.populate_initial_values()  # Populate with initial values
        
        
    def clear_figure(self):
        """Clear the figure in the plot area."""
        self.ax.clear()
        self.ax.set_title("Model Parameters", fontsize=16)  # Optional: Reset title to default
        self.ax.set_ylabel("Value", fontsize=12)
        self.canvas.draw()  # Refresh the canvas

    def load_new_build(self):
        """Load parameters for New Build House."""
        building_type = 'new_build_house'  # Specify building type
        
        # Clear all current values
        self.clear_input_values()
        
        # Extract data from YAML
        roof_U_value = building_data[building_type]['roof_U_value']['value']
        roof_area = building_data[building_type]['roof_area']['value']
        wall_U_value = building_data[building_type]['wall_U_value']['value']
        wall_area = building_data[building_type]['wall_area']['value']
        
        mass_of_water = "200"  # Example value
        initial_temp = "318"  # Example value
        heat_pump_on = "313"  # Example value
        heat_pump_off = "333"  # Example value
        heat_loss_coeff = "5"  # Example value
        heat_transfer_coeff = "300"  # Example value
        
        # Populate input fields
        self.entries["Roof U Value"].insert(0, roof_U_value)
        self.entries["Roof Area"].insert(0, roof_area)
        self.entries["Wall U Value"].insert(0, wall_U_value)
        self.entries["Wall Area"].insert(0, wall_area)
        self.entries["Mass of Water in Hot Water Tank"].insert(0, mass_of_water)
        self.entries["Initial Tank Temperature"].insert(0, initial_temp)
        self.entries["Heat Pump On"].insert(0, heat_pump_on)
        self.entries["Heat Pump Off"].insert(0, heat_pump_off)
        self.entries["Heat Loss Coefficient"].insert(0, heat_loss_coeff)
        self.entries["Heat Transfer Coefficient"].insert(0, heat_transfer_coeff)

    def load_larch_lecture(self):
        """Load parameters for Larch Lecture Theatre."""
        building_type = 'larch_Lecture_theatre'  # Specify building type
        
        # Clear all current values
        self.clear_input_values()
        
        # Extract data from YAML
        roof_U_value = building_data[building_type]['roof_U_value']['value']
        roof_area = building_data[building_type]['roof_area']['value']
        wall_U_value = building_data[building_type]['wall_U_value']['value']
        wall_area = building_data[building_type]['wall_area']['value']
        
        mass_of_water = "200"  # Example value
        initial_temp = "318"  # Example value
        heat_pump_on = "313"  # Example value
        heat_pump_off = "333"  # Example value
        heat_loss_coeff = "5"  # Example value
        heat_transfer_coeff = "300"  # Example value
        
        # Populate input fields
        self.entries["Roof U Value"].insert(0, roof_U_value)
        self.entries["Roof Area"].insert(0, roof_area)
        self.entries["Wall U Value"].insert(0, wall_U_value)
        self.entries["Wall Area"].insert(0, wall_area)
        self.entries["Mass of Water in Hot Water Tank"].insert(0, mass_of_water)
        self.entries["Initial Tank Temperature"].insert(0, initial_temp)
        self.entries["Heat Pump On"].insert(0, heat_pump_on)
        self.entries["Heat Pump Off"].insert(0, heat_pump_off)
        self.entries["Heat Loss Coefficient"].insert(0, heat_loss_coeff)
        self.entries["Heat Transfer Coefficient"].insert(0, heat_transfer_coeff)

    def load_warehouse(self):
        """Load parameters for Warehouse."""
        building_type = 'warehouse'  # Specify building type
        
        # Clear all current values
        self.clear_input_values()
        
        # Extract data from YAML
        roof_U_value = building_data[building_type]['roof_U_value']['value']
        roof_area = building_data[building_type]['roof_area']['value']
        wall_area = building_data[building_type]['wall_area']['value']
        wall_U_value = building_data[building_type]['wall_U_value']['value']
        
        # Example values for other parameters
        mass_of_water = "200"  # Example value
        initial_temp = "318"  # Example value
        heat_pump_on = "313"  # Example value
        heat_pump_off = "333"  # Example value
        heat_loss_coeff = "5"  # Example value
        heat_transfer_coeff = "300"  # Example value
        
        # Populate input fields
        self.entries["Roof U Value"].insert(0, roof_U_value)
        self.entries["Roof Area"].insert(0, roof_area)
        self.entries["Wall U Value"].insert(0, wall_U_value)
        self.entries["Wall Area"].insert(0, wall_area)
        self.entries["Mass of Water in Hot Water Tank"].insert(0, mass_of_water)
        self.entries["Initial Tank Temperature"].insert(0, initial_temp)
        self.entries["Heat Pump On"].insert(0, heat_pump_on)
        self.entries["Heat Pump Off"].insert(0, heat_pump_off)
        self.entries["Heat Loss Coefficient"].insert(0, heat_loss_coeff)
        self.entries["Heat Transfer Coefficient"].insert(0, heat_transfer_coeff)

    def clear_input_values(self):
        """Clear all input values in the entry fields."""
        for entry in self.entries.values():
            entry.delete(0, 'end')  # Clear each entry

    def generate_gradient_colors(self, steps=20):
        """Generate a list of colors representing a gradient from blue to yellow to red."""
        blue = np.array([0, 120, 215]) / 255
        yellow = np.array([1, 1, 0])
        red = np.array([1, 0, 0])

        colors = []

        # First gradient: Blue at 10°C to Yellow at 15°C
        for i in range(steps // 2):
            color = blue + (yellow - blue) * (i / (steps // 2 - 1))
            colors.append(self.rgb_to_hex(color))

        # Second gradient: Yellow at 15°C to Red at 30°C
        for i in range(steps // 2):
            color = yellow + (red - yellow) * (i / (steps // 2 - 1))
            colors.append(self.rgb_to_hex(color))

        return colors

    def rgb_to_hex(self, rgb):
        """Convert RGB values to hexadecimal color format."""
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def update_slider_value(self, value):
        """Update the slider value label and change color based on the slider value."""
        self.current_value = float(value)
        self.current_value_label.configure(text=f"{self.current_value:.1f} °C")

        # Change color based on the value of the slider
        if self.current_value < 10:
            color_index = 0  # Blue
            text_color = "#0078D7"  # Original blue color
        elif 10 <= self.current_value < 15:
            color_index = int((self.current_value - 10) / 5 * 10)  # Blue to Yellow
            text_color = self.generate_gradient_colors()[color_index]
        elif 15 <= self.current_value < 30:
            color_index = int(10 + (self.current_value - 15) / 15 * 10)  # Yellow to Red
            text_color = self.generate_gradient_colors()[color_index]
        else:
            color_index = 19  # Red
            text_color = "red"

        colors = self.generate_gradient_colors()
        self.indoor_temp_slider.configure(fg_color=colors[color_index])
        self.current_value_label.configure(text_color=text_color)

    def heat(self):
        print("heat transfer button pressed.")
        
        day = int(self.day_entry.get())
        month = int(self.month_entry.get())
        year = int(self.year_entry.get())
        print(f"Execute function: Day={day}, Month={month}, Year={year}")
        
        values = []
        for entry in self.entries.values():
            value = entry.get()
            if value.replace('.', '', 1).isdigit():  # Allow one decimal point
                values.append(float(value))
            else:
                ctk.CTkMessageBox.showerror("Input Error", f"Invalid value for {entry}: {value}")
                return  # Exit if there's an error
    
        # Append the current slider value to the list
        values.append(self.current_value)
        
        # Define the location (Edinburgh: 55.9533° N, 3.1883° W)
        location = Point(55.9533, -3.1883)

        # Set the time period for data retrieval
        start = datetime(year, month, day, 0) # Start time
        end = start + timedelta(days=1)  # End time (24-hour period)

        # Fetch hourly temperature data
        data = Hourly(location,start,end)
        data = data.fetch()
        data = data.reset_index()
        t_ambient = []
        hours = data["time"].dt.hour
        for i in range(0,25):
            t = data["temp"].iloc[i] + 273 
            t_ambient.append(t)
        
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
        
        
        indoor_setpoint_temperature = values[10]+273
        print(indoor_setpoint_temperature)
        roof_U_value = values[0]
        roof_area = values[1]
        wall_U_value = values[2]
        wall_area = values[3]

        fixed_condenser_temperature = data_inputs['heat_pump']['fixed_condenser_temperature_K']['value']
        heat_transfer_area = data_inputs['heat_pump']['heat_transfer_area']['value']
        off_temperature_threshold = values[7]
        on_temperature_threshold = values[6]
        overall_heat_transfer_coefficient = values[9]

        heat_loss_coefficient = values[8]
        mass_of_water = values[4]
        specific_heat_capacity = data_inputs['hot_water_tank']['specific_heat_capacity']['value']
        total_thermal_capacity = data_inputs['hot_water_tank']['total_thermal_capacity']['value']

        initial_tank_temperature = values[5]

        time_step = data_inputs['simulation_parameters']['time_points']['value']
        total_time = data_inputs['simulation_parameters']['total_time_seconds']['value']
        
        
        
        delta_t_ambient =  [temp - indoor_setpoint_temperature for temp in t_ambient]



        Q_load = np.abs([((wall_area * wall_U_value * t) + (roof_area * roof_U_value * t)) for t in delta_t_ambient])
        
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
        
        
        tank_surface_area = 2
        
        
        # Initialize the Heat transfer, load and loss lists
        current_list = []
        loss_list = []
        load_list = []
        transfer_list = []
        time_hours = time_array / 3600
        
        def tank_temperature_ode(t, y, heat_pump_status):
                t_tank = y[0]  # Unpack the tank temperature from the state vector
                
                current_hour = int(t // 3600)
                
                # Assign the current ambient temperature from the list of hours
                t_ambient_current = t_ambient[current_hour]
                delta_t_ambient_current = t_ambient_current - indoor_setpoint_temperature
                
                current_list.append(t_ambient_current)
                
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
         
         
        # Plotting all three datasets on the same graph
        self.ax.clear()
        self.ax.plot(time_hours[:-1], loss_list, label='Heat Loss (W)', color='red')  
        self.ax.plot(time_hours[:-1], transfer_list, label='Heat Transfer (W)', color='blue')  
        self.ax.plot(time_hours[:-1], load_list, label='Heat Load (W)', color='green')  
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Power (W)')
        self.ax.set_title('Heat Loss, Transfer, and Load Over Time')
        self.ax.legend(loc = 'upper right')
        self.ax.grid(True)
        self.canvas.draw()


    def execute(self):
        """Execute button functionality."""
        print("Execute button pressed.")
        
        day = int(self.day_entry.get())
        month = int(self.month_entry.get())
        year = int(self.year_entry.get())
        print(f"Execute function: Day={day}, Month={month}, Year={year}")
        
        values = []
        for entry in self.entries.values():
            value = entry.get()
            if value.replace('.', '', 1).isdigit():  # Allow one decimal point
                values.append(float(value))
            else:
                ctk.CTkMessageBox.showerror("Input Error", f"Invalid value for {entry}: {value}")
                return  # Exit if there's an error
    
        # Append the current slider value to the list
        values.append(self.current_value)
        
        # Define the location (Edinburgh: 55.9533° N, 3.1883° W)
        location = Point(55.9533, -3.1883)

        # Set the time period for data retrieval
        start = datetime(year, month, day, 0) # Start time
        end = start + timedelta(days=1)  # End time (24-hour period)

        # Fetch hourly temperature data
        data = Hourly(location,start,end)
        data = data.fetch()
        data = data.reset_index()
        t_ambient = []
        hours = data["time"].dt.hour
        for i in range(0,25):
            t = data["temp"].iloc[i] + 273 
            t_ambient.append(t)
        
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
        
        
        indoor_setpoint_temperature = values[10]+273
        print(indoor_setpoint_temperature)
        roof_U_value = values[0]
        roof_area = values[1]
        wall_U_value = values[2]
        wall_area = values[3]

        fixed_condenser_temperature = data_inputs['heat_pump']['fixed_condenser_temperature_K']['value']
        heat_transfer_area = data_inputs['heat_pump']['heat_transfer_area']['value']
        off_temperature_threshold = values[7]
        on_temperature_threshold = values[6]
        overall_heat_transfer_coefficient = values[9]

        heat_loss_coefficient = values[8]
        mass_of_water = values[4]
        specific_heat_capacity = data_inputs['hot_water_tank']['specific_heat_capacity']['value']
        total_thermal_capacity = data_inputs['hot_water_tank']['total_thermal_capacity']['value']

        initial_tank_temperature = values[5]

        time_step = data_inputs['simulation_parameters']['time_points']['value']
        total_time = data_inputs['simulation_parameters']['total_time_seconds']['value']
        
        
        
        delta_t_ambient =  [temp - indoor_setpoint_temperature for temp in t_ambient]



        Q_load = np.abs([((wall_area * wall_U_value * t) + (roof_area * roof_U_value * t)) for t in delta_t_ambient])
        
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
        
        
        tank_surface_area = 2
        
        
        # Initialize the Heat transfer, load and loss lists
        current_list = []
        loss_list = []
        load_list = []
        transfer_list = []
        time_hours = time_array / 3600
        
        def tank_temperature_ode(t, y, heat_pump_status):
                t_tank = y[0]  # Unpack the tank temperature from the state vector
                
                current_hour = int(t // 3600)
                
                # Assign the current ambient temperature from the list of hours
                t_ambient_current = t_ambient[current_hour]
                delta_t_ambient_current = t_ambient_current - indoor_setpoint_temperature
                
                current_list.append(t_ambient_current)
                
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
        
        print(values)
        
        self.ax.clear()
        self.ax.plot(time_hours, T_tank_solution, label='Tank Temperature (K)')
        self.ax.set_xlabel('Time (hours)')
        self.ax.set_ylabel('Tank Temperature (K)')
        self.ax.set_title('Tank Temperature Over Time')

        self.ax.set_xlim(0, 24)
        self.ax.grid(True)
        self.ax.legend(loc = 'upper right')
        self.canvas.draw()
    
    


# Run the GUI application
if __name__ == "__main__":
    GUI()
