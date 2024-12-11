from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.station import Station

class Bike:

    def __init__(self, bike_id: int, stn: "Station" = None, max_battery: float = 50.0):
        """
        Initialize a Bike object.

        Parameters:
        bike_id (int): Unique identifier for the bike.
        stn (Station): The station where the bike is located.
        max_battery (int): Maximum battery capacity of the bike. Default is 100 (in km).
        """
        self.bike_id = bike_id
        self.stn = stn
        self.max_battery = max_battery
        self.battery = max_battery
        self.available = True

    def __str__(self):
        """
        Return a string representation of the Bike object.

        Returns:
        str: A string describing the bike with its ID and current station.
        """
        return f"Bike {self.bike_id} at {self.stn} - Battery: {self.battery} km - Available: {self.available}"

    def set_availability(self, available: bool):
        """
        Set the availability status of the bike.

        Parameters:
        available (bool): True if the bike is available, False otherwise.
        """
        self.available = available

    def set_station(self, stn: "Station"):
        """
        Set the station where the bike is located.

        Parameters:
        stn (Station): The station where the bike is located.
        """
        self.stn = stn

    def set_battery(self, battery: float):
        """
        Set the battery level of the bike.

        Parameters:
        battery (int): The battery level of the bike.
        """
        self.battery = battery

    def get_station(self) -> "Station":
        """
        Get the station where the bike is located.

        Returns:
        Station: The station where the bike is located.
        """
        return self.stn

    def get_battery(self) -> float:
        """
        Get the battery level of the bike.

        Returns:
        int: The battery level of the bike.
        """
        return self.battery

    def get_bike_id(self) -> int:
        """
        Get the ID of the bike.

        Returns:
        int: The ID of the bike.
        """
        return self.bike_id

    def get_availability(self) -> bool:
        """
        Get the availability status of the bike.

        Returns:
        bool: True if the bike is available, False otherwise.
        """
        return self.available

    def get_max_battery(self) -> float:
        """
        Get the maximum battery capacity of the bike.

        Returns:
        float: The maximum battery capacity of the bike.
        """
        return self.max_battery