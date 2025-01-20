from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.bike import Bike
    from gymnasium_env.simulator.cell import Cell

class Station:
    def __init__(self, station_id: int, lat: float, lon: float, name: str = None, capacity: int = 1000,
                 bikes: {"Bike"} = None, request_rate: float = 0.0, cell: "Cell" = None):
        """
        Initialize a Station object.

        Parameters:
        station_id (int): Unique identifier for the station.
        name (str): Name of the station.
        lat (float): Latitude of the station's location.
        lon (float): Longitude of the station's location.
        capacity (int): Maximum capacity of bikes the station can hold. Default is 1000.
        bikes (list): List of Bike objects at the station. Default is an empty list.
        request_rate (float): Rate of bike requests at the station. Default is 0.0.
        """
        self.station_id = station_id
        self.name = name if name is not None else f"Station {station_id}"
        self.lat = lat
        self.lon = lon
        self.capacity = capacity
        self.bikes = bikes if bikes is not None else {}
        self.request_rate = request_rate
        self.cell = cell
        self.number_of_bikes = len(self.bikes)

    def __str__(self):
        """
        Return a string representation of the Station object.

        Returns:
        str: A string describing the station with its ID, name, latitude, and longitude.
        """
        return f"Station {self.station_id}: ({self.lat}, {self.lon})"

    def set_bikes(self, bikes: {"Bike"}):
        """
        Set the list of bikes at the station.

        Parameters:
        bikes (list): List of Bike objects at the station.
        """
        if len(bikes) < self.capacity:
            for bike_id, bike in bikes.items():
                bike.set_availability(True)
                bike.set_station(self)
                self.bikes[bike_id] = bike
                self.cell.set_total_bikes(self.cell.get_total_bikes() + 1)
        else:
            raise ValueError("The number of bikes exceeds the station's capacity.")

    def set_request_rate(self, request_rate: float):
        """
        Set the request rate of the station.

        Parameters:
        request_rate (float): The rate of bike requests at the station.
        """
        self.request_rate = request_rate

    def set_capacity(self, capacity: int):
        """
        Set the capacity of the station.

        Parameters:
        capacity (int): The maximum capacity of bikes the station can hold.
        """
        self.capacity = capacity

    def set_cell(self, cell: "Cell"):
        """
        Set the cell of the station.

        Parameters:
        cell (Cell): The cell object representing the station's location.
        """
        self.cell = cell

    def unlock_bike(self, bike_id: int = None) -> "Bike":
        """
        Unlock a bike from the station.

        Returns:
        Bike: The bike unlocked from the station.
        """
        if len(self.bikes) > 0:
            if bike_id is None:
                bike_id = max(self.bikes, key=lambda k: self.bikes[k].get_battery())
            bike = self.bikes.pop(bike_id)
            bike.set_availability(False)
            bike.set_station(None)
            self.cell.set_total_bikes(self.cell.get_total_bikes() - 1)
            return bike
        else:
            raise ValueError("Station is empty. Cannot unlock bike.")

    def lock_bike(self, bike: "Bike"):
        """
        Lock a bike at the station.

        Parameters:
        bike (Bike): The bike to be locked at the station.
        """
        if len(self.bikes) < self.capacity:
            bike.set_availability(True)
            bike.set_station(self)
            self.bikes[bike.get_bike_id()] = bike
            self.cell.set_total_bikes(self.cell.get_total_bikes() + 1)
        else:
            raise ValueError("Station is full. Cannot lock bike. Dimension: " + str(len(self.bikes)) + "/" + str(self.capacity))

    def get_station_id(self) -> int:
        """
        Get the ID of the station.

        Returns:
        int: The ID of the station.
        """
        return self.station_id

    def get_name(self) -> str:
        """
        Get the name of the station.

        Returns:
        str: The name of the station.
        """
        return self.name

    def get_coordinates(self) -> (float, float):
        """
        Get the coordinates of the station.

        Returns:
        tuple: A tuple containing the latitude and longitude of the station.
        """
        return self.lat, self.lon

    def get_bikes(self) -> {"Bike"}:
        """
        Get the list of bikes at the station.

        Returns:
        list: A list of Bike objects at the station.
        """
        return self.bikes

    def get_request_rate(self) -> float:
        """
        Get the request rate of the station.

        Returns:
        float: The rate of bike requests at the station.
        """
        return self.request_rate

    def get_cell(self) -> "Cell":
        """
        Get the cell of the station.

        Returns:
        Cell: The cell object representing the station's location.
        """
        return self.cell

    def get_capacity(self) -> int:
        """
        Get the capacity of the station.

        Returns:
        int: The maximum capacity of bikes the station can hold.
        """
        return self.capacity

    def get_number_of_bikes(self) -> int:
        """
        Get the number of bikes at the station.

        Returns:
        int: The number of bikes at the station.
        """
        return len(self.bikes)