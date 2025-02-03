from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gymnasium_env.simulator.bike import Bike
    from gymnasium_env.simulator.cell import Cell

class Truck:
    truck_id = 0
    def __init__(self, position: int, cell: "Cell", bikes: dict[int, "Bike"], max_range: float = 300, max_load: int = 30):
        self.id = Truck.truck_id
        self.position = position
        self.cell = cell
        self.max_range = max_range
        self.range = max_range
        self.max_load = max_load
        self.bikes = bikes.copy()
        self.current_load = len(bikes) if bikes is not None else 0
        self.leaving_cell = cell
        self.last_charge = 0

        Truck.truck_id += 1

    def __str__(self):
        return f"Truck {self.id} at {self.position} - Range: {self.range} km - Load: {self.current_load} bikes."

    def set_position(self, position: int):
        self.position = position

    def set_cell(self, cell: "Cell"):
        self.leaving_cell = self.cell
        self.cell = cell

    def set_range(self, r: float):
        self.range = r

    def set_load(self, bikes: dict[int, "Bike"]):
        self.current_load = len(bikes)
        self.bikes = bikes.copy()

    def load_bike(self, bike: "Bike"):
        if self.current_load < self.max_load:
            bike.set_station(None)
            bike.set_availability(False)
            max_battery = bike.get_max_battery()
            self.last_charge = (max_battery - bike.get_battery()) / max_battery
            bike.set_battery(max_battery)
            self.bikes[bike.bike_id] = bike
            self.current_load += 1
        else:
            raise ValueError("Truck is full")

    def unload_bike(self) -> "Bike":
        if self.current_load > 0:
            bike = self.bikes.pop(next(iter(self.bikes)))
            self.current_load -= 1
            return bike
        else:
            raise ValueError("Truck is empty")

    def get_position(self) -> int:
        return self.position

    def get_cell(self) -> "Cell":
        return self.cell

    def get_leaving_cell(self) -> "Cell":
        return self.leaving_cell

    def get_range(self) -> float:
        return self.range

    def get_load(self) -> int:
        return self.current_load