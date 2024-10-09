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

#Creates a list for ambient temperature values and a list of the hours 
t_ambient = []
hours = data["time"].dt.hour

for i in range(0,25):
    t = data["temp"].iloc[i] + 273 
    t_ambient.append(t)
    

