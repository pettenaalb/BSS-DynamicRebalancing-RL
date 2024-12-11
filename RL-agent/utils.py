import networkx as nx
import torch

from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch, Data

def convert_graph_to_data(graph: nx.MultiDiGraph):
    data = from_networkx(graph)

    # Extract node attributes
    node_attrs = [
        'demand_rate_per_region',
        'average_battery_level_per_region',
        'low_battery_ratio_per_region',
        'variance_battery_level_per_region',
        'total_bikes_per_region'
    ]
    data.x = torch.cat([
        torch.tensor([graph.nodes[n].get(attr, 0) for n in graph.nodes()], dtype=torch.float).unsqueeze(-1) for attr in node_attrs
    ], dim=-1)

    # Extract edge types
    edge_types = []
    edge_attrs = ['distance']
    edge_attr_list = {attr: [] for attr in edge_attrs}
    for u, v, k, attr in graph.edges(data=True, keys=True):
        edge_types.append(k)
        for edge_attr in edge_attrs:
            edge_attr_list[edge_attr].append(attr[edge_attr])

    # Map edge types to integers
    edge_type_mapping = {etype: i for i, etype in enumerate(set(edge_types))}
    edge_type_indices = torch.tensor([edge_type_mapping[etype] for etype in edge_types],
                                     dtype=torch.long)

    # Add edge types and attributes to the data object
    data.edge_type = edge_type_indices
    data.edge_attr = torch.cat([torch.tensor(edge_attr_list[attr],
                                             dtype=torch.float).view(-1, 1) for attr in edge_attrs],
                               dim=-1)
    data.edge_index = data.edge_index

    return data


def move_to_device(data, device):
    """
    Recursively moves data (tensors, lists, dictionaries) to the specified device.

    Parameters:
        - data: Data to move (tensor, dict, list, Batch, etc.).
        - device: Target device (e.g., 'cuda' or 'cpu').

    Returns:
        - Data moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Batch):  # Handle PyTorch Geometric Batch
        return data.to(device)
    elif isinstance(data, dict):  # Recursively move dictionary contents
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):  # Recursively move list elements
        return [move_to_device(item, device) for item in data]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def cast_to_float32(data):
    # Convert PyTorch Geometric Data object
    if isinstance(data, Data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.dtype in [torch.float64, torch.float16]:
                data[key] = value.float()
        return data

    # Convert standard PyTorch tensor
    elif isinstance(data, torch.Tensor) and data.dtype in [torch.float64, torch.float16]:
        return data.float()

    # Recursively process dicts
    elif isinstance(data, dict):
        return {k: cast_to_float32(v) for k, v in data.items()}

    # Return as is for other types
    return data


def convert_seconds_to_hours_minutes(seconds) -> str:
    hours, remainder = divmod(seconds, 3600)
    hours = hours % 24
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"