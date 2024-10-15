import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import requests
import yaml

# URLs to the YAML files
heat_pump_cop_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/heat_pump_cop_synthetic_full.yaml"
inputs_file = "https://raw.githubusercontent.com/erikasdaniels/CMM3-Group-24-Project/refs/heads/main/inputs.yaml"

# Fetch the YAML data
response_cop = requests.get(heat_pump_cop_file)
response_inputs = requests.get(inputs_file)

if response_cop.status_code == 200:
    data_cop = yaml.safe_load(response_cop.text)

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

class GUI:
    
    def __init__(self):
        # Configure the main window
        ctk.set_appearance_mode("dark")  # Options: "dark", "light", or "system"
        ctk.set_default_color_theme("blue")  # Change this to other themes as needed

        self.root = ctk.CTk()  # Use CTk instead of Tk
        
        # Set the geometry to a reasonable size
        self.root.geometry("1200x800")  # Width x Height
        
        self.root.title("Task B - User Interface - Group 24")

        # Title label (updated text, font, and formatting)
        self.label = ctk.CTkLabel(self.root, text='Heat Pump Model - Group 24', font=("Arial", 48, 'underline'))
        self.label.pack(pady=(20, 20))  # Increased top padding for spacing

        # Frame for input boxes on the left
        input_frame = ctk.CTkFrame(self.root)
        input_frame.pack(side="left", padx=(40, 10), pady=20)  # Adjusted padx for input frame

        # Adjusting the Heating label above the slider
        adjusting_label = ctk.CTkLabel(input_frame, text='Adjusting The Heating', font=("Arial", 18, 'underline'))
        adjusting_label.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="W")  # Left-aligned
        
        # Slider for Indoor Temperature Set Point (changed range to 10-30)
        self.indoor_temp_label = ctk.CTkLabel(input_frame, text='Indoor Temperature Set Point (°C):')
        self.indoor_temp_label.grid(row=1, column=0, sticky="W", pady=(10, 5))  # Positioned above parameters
        
        self.indoor_temp_slider = ctk.CTkSlider(input_frame, from_=10, to=30, number_of_steps=20)  # Adjusted range
        self.indoor_temp_slider.set(23)  # Set initial value
        self.indoor_temp_slider.grid(row=2, column=0, pady=(0, 5), sticky="W")  # Left-aligned slider

        # Initialize the current value label below the slider
        self.current_value = self.indoor_temp_slider.get()  # Get initial slider value
        self.current_value_label = ctk.CTkLabel(input_frame, text=f"{self.current_value:.1f} °C", text_color="#0078D7")  # Blue color
        self.current_value_label.grid(row=4, column=0, pady=(0, 20), sticky="W")  # Align with current value text

        # Update the slider value label when the slider changes
        self.indoor_temp_slider.configure(command=self.update_slider_value)

        # Update the slider value and color initially based on the set point of 23°C
        self.update_slider_value(self.indoor_temp_slider.get())  # Initialize color based on the initial value

        # Model Parameters label
        model_parameters_label = ctk.CTkLabel(input_frame, text='Model Parameters', font=("Arial", 18, 'underline'))
        model_parameters_label.grid(row=5, column=0, columnspan=2, pady=(0, 10), sticky="W")  # Left-aligned

        # Input labels and entry fields with vertical spacing
        inputs = [
            ("Roof U Value", "W/m²K", roof_U_value),  # Separate Roof U Value
            ("Roof Area", "m²", roof_area),            # Separate Roof Area
            ("Wall U Value", "W/m²K", wall_U_value),  # Separate Wall U Value
            ("Wall Area", "m²", wall_area),            # Separate Wall Area
            ("Mass of Water in Hot Water Tank", "kg", mass_of_water),
            ("Initial Tank Temperature", "°C", initial_tank_temperature - 273.15),  # Convert K to °C
            ("Heat Pump On", "°C", on_temperature_threshold - 273.15),  # Convert K to °C
            ("Heat Pump Off", "°C", off_temperature_threshold - 273.15),  # Convert K to °C
            ("Heat Loss Coefficient", "W/K", heat_loss_coefficient),
            ("Heat Transfer Coefficient", "W/K", overall_heat_transfer_coefficient),
        ]
        
        self.entries = {}
        for i, (label_text, unit, initial_value) in enumerate(inputs):
            label = ctk.CTkLabel(input_frame, text=f"{label_text} ({unit}):")
            label.grid(row=i + 6, column=0, sticky="W", pady=(5, 5))  # Adjusted row index for positioning
            
            entry = ctk.CTkEntry(input_frame)
            entry.grid(row=i + 6, column=1, pady=(5, 5), padx=(10, 0))  # Adjusted row index for positioning
            
            # Set the initial value for the entry if available
            if initial_value is not None:
                entry.insert(0, str(initial_value))  # Convert initial value to string for entry
            
            self.entries[label_text] = entry
        
        # Execute button at the bottom of input boxes, with a sensible buffer
        self.execute_button = ctk.CTkButton(input_frame, text="Execute", command=self.execute, width=150)
        self.execute_button.grid(row=len(inputs) + 6, column=0, columnspan=2, pady=10)

        # Clear button functionality
        self.clear_button = ctk.CTkButton(input_frame, text="Clear Figure", command=self.clear_figure, width=150)
        self.clear_button.grid(row=len(inputs) + 7, column=0, columnspan=2, pady=(5, 20))

        # Create matplotlib figure on the right
        figure_frame = ctk.CTkFrame(self.root)
        figure_frame.pack(side="right", fill="both", expand=True, padx=10, pady=20)
        
        # Set the figure size to maintain a 4:3 aspect ratio
        self.fig = Figure(figsize=(10.67, 8), dpi=100)  # 10.67:8 ratio (4:3)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Variable to keep track of button state
        self.is_clearing = False

        self.root.mainloop()

    def generate_gradient_colors(self, steps=20):
        """Generate a list of colors representing a gradient from blue to yellow to red."""
        blue = np.array([0, 120, 215]) / 255  # RGB for original blue
        yellow = np.array([1, 1, 0])  # RGB for yellow
        red = np.array([1, 0, 0])  # RGB for red

        colors = []

        # First gradient: Blue at 10°C to Yellow at 15°C
        for i in range(steps // 2):
            # Linear interpolation
            color = blue + (yellow - blue) * (i / (steps // 2 - 1))
            colors.append(self.rgb_to_hex(color))

        # Second gradient: Yellow at 15°C to Red at 30°C
        for i in range(steps // 2):
            # Linear interpolation
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
            text_color = self.generate_gradient_colors()[color_index]  # Gradient text color
        elif 15 <= self.current_value < 30:
            color_index = int(10 + (self.current_value - 15) / 15 * 10)  # Yellow to Red
            text_color = self.generate_gradient_colors()[color_index]  # Gradient text color
        else:
            color_index = 19  # Red
            text_color = "red"  # Red text color

        colors = self.generate_gradient_colors()
        self.indoor_temp_slider.configure(fg_color=colors[color_index])
        self.current_value_label.configure(text_color=text_color)  # Change text color based on temperature

    def execute(self):
        """Placeholder for the execute button functionality."""
        print("Execute button pressed.")
        # Example data for bar chart (for demonstration purposes)
        categories = list(self.entries.keys())
        values = [float(entry.get()) for entry in self.entries.values() if entry.get().replace('.', '', 1).isdigit()]
        
        # Clear previous bars
        self.ax.clear()

        # Create the bar chart
        self.ax.bar(categories, values, color='skyblue')
        self.ax.set_title("Model Parameters", fontsize=16)
        self.ax.set_ylabel("Value", fontsize=12)
        self.ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Redraw the canvas to show the new chart
        self.canvas.draw()

    def clear_figure(self):
        """Clear the figure in the bar chart."""
        self.ax.clear()
        self.ax.set_title("Model Parameters", fontsize=16)
        self.ax.set_ylabel("Value", fontsize=12)
        self.canvas.draw()  # Refresh the canvas

# Run the GUI application
if __name__ == "__main__":
    GUI()
