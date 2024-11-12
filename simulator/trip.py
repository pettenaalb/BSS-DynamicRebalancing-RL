from station import Station
from bike import Bike
from utils import convert_seconds_to_hours_minutes

class Trip:

    trip_id = 0

    def __init__(self, start_time: int, end_time: int, start_location: Station, end_location: Station, bike: Bike = None,
                 distance: int = 0, failed = False, deviated = False, deviated_location = None):
        """
        Initialize a Trip object.

        Parameters:
        start_time (int): The time the trip started.
        end_time (int): The time the trip ended.
        start_location (tuple): The starting location of the trip.
        end_location (tuple): The ending location of the trip.
        vehicle_id (int): The ID of the vehicle used for the trip.
        driver_id (int): The ID of the driver for the trip.
        """
        self.trip_id = Trip.trip_id
        self.start_time = start_time
        self.end_time = end_time
        self.start_location = start_location
        self.end_location = end_location
        self.bike = bike
        self.distance = distance
        self.failed = failed
        self.deviated = deviated
        self.deviated_location = deviated_location

        Trip.trip_id += 1

    def __str__(self):
        """
        Return a string representation of the Trip object.
        """
        if self.deviated:
            return (f"Trip {self.trip_id}: {self.start_location} to {self.end_location} - Bike {self.bike.bike_id} "
                    f"- Time: {convert_seconds_to_hours_minutes(self.start_time)} to {convert_seconds_to_hours_minutes(self.end_time)} "
                    f"- Starting station deviated to {self.deviated_location}")

        return (f"Trip {self.trip_id}: {self.start_location} to {self.end_location} - Bike {self.bike.bike_id} "
                f"- Time: {convert_seconds_to_hours_minutes(self.start_time)} to {convert_seconds_to_hours_minutes(self.end_time)}")

    def set_bike(self, bike: Bike):
        self.bike = bike

    def set_failed(self, failed: bool):
        self.failed = failed

    def set_deviated(self, deviated: bool):
        self.deviated = deviated

    def set_deviated_location(self, deviated_location: Station):
        self.deviated_location = deviated_location

    def is_failed(self):
        return self.failed

    def is_deviated(self):
        return self.deviated

    def get_trip_id(self):
        return self.trip_id

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_start_location(self):
        return self.start_location

    def get_end_location(self):
        return self.end_location

    def get_bike(self):
        return self.bike

    def get_distance(self):
        return self.distance

    def get_deviated_location(self):
        return self.deviated_location