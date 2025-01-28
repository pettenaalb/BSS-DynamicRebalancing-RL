import osmnx as ox
import networkx as nx
import ast
import math

from geopy.distance import geodesic
from shapely.geometry import Polygon, Point
from shapely import wkt

class Cell:
    def __init__(self, cell_id, boundary: Polygon):
        self.id = cell_id
        self.boundary = boundary
        self.nodes = []
        self.center_node = 0
        self.diagonal = 0
        self.adjacent_cells = {'up': None, 'down': None, 'left': None, 'right': None}
        self.total_bikes = 0
        self.capacity = 0
        self.request_rate = 0
        self.visits = 0
        self.critic_score = 0
        self.is_critical = False

    def set_center_node(self, graph: nx.MultiDiGraph):
        center_coords = self.boundary.centroid.coords[0]
        nearest_node = ox.distance.nearest_nodes(graph, center_coords[0], center_coords[1])
        if nearest_node in self.nodes:
            self.center_node = nearest_node
        else:
            raise ValueError("Center node not found in cell nodes")

    def contain_nodes (self, point: Point) -> bool:
        return self.boundary.contains(point)

    def to_dict(self):
        # Serialize boundary as WKT (Well-Known Text) and other attributes
        return {
            'id': self.id,
            'boundary': self.boundary.wkt,  # Convert boundary to WKT string for saving
            'nodes': ','.join(map(str, self.nodes)),  # Store nodes as a comma-separated string
            'center_node': self.center_node,
            'diagonal': self.diagonal,
            'adjacent_cells': self.adjacent_cells,
        }

    @classmethod
    def from_dict(cls, data):
        # Convert WKT boundary back to a Polygon, and nodes from a comma-separated string
        boundary = wkt.loads(data['boundary'])
        cell = cls(cell_id=data['id'], boundary=boundary)
        cell.nodes = list(map(int, data['nodes'].split(',')))
        cell.center_node = data['center_node']
        cell.diagonal = data['diagonal']
        cell.adjacent_cells = ast.literal_eval(data['adjacent_cells'])
        return cell

    def set_diagonal(self):
        coords = list(self.boundary.exterior.coords)[:-1]
        side_length_meters = geodesic(coords[0], coords[1]).meters
        self.diagonal = int(math.sqrt(2) * side_length_meters)

    def set_total_bikes(self, total_bikes: int):
        self.total_bikes = total_bikes

    def set_capacity(self, capacity: int):
        self.capacity = capacity

    def set_request_rate(self, request_rate: float):
        self.request_rate = request_rate

    def set_visits(self, visits: int):
        self.visits = visits

    def set_critic_score(self, critic_score: float):
        self.critic_score = critic_score
        if critic_score < 0.5:
            self.is_critical = True
        else:
            self.is_critical = False

    def get_id(self) -> int:
        return self.id

    def get_boundary(self) -> Polygon:
        return self.boundary

    def get_nodes(self) -> list[int]:
        return self.nodes

    def get_center_node(self) -> int:
        return self.center_node

    def get_adjacent_cells(self) -> dict:
        return self.adjacent_cells

    def get_diagonal(self) -> int:
        return self.diagonal

    def get_total_bikes(self) -> int:
        return self.total_bikes

    def get_capacity(self) -> int:
        return self.capacity

    def reset(self):
        self.total_bikes = 0

    def get_request_rate(self) -> float:
        return self.request_rate

    def get_visits(self) -> int:
        return self.visits

    def get_critic_score(self) -> float:
        return self.critic_score

    def is_critical(self) -> bool:
        return self.is_critical
