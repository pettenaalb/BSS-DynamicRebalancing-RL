import numpy as np

def generate_poisson_events(rate, time_duration):
    num_events = np.random.poisson(rate * time_duration)
    inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
    event_times = np.cumsum(inter_arrival_times).astype(np.int64)
    return num_events, event_times, inter_arrival_times

def poisson_simulation(rate, time_duration):
    if isinstance(rate, float):
        if rate != 0:
            num_events, event_times, inter_arrival_times = generate_poisson_events(rate, time_duration)
        else:
            num_events, event_times, inter_arrival_times = None, None, None
        return num_events, event_times, inter_arrival_times

    elif isinstance(rate, list):
        print("entro qua2")
        num_events_list = []
        event_times_list = []
        inter_arrival_times_list = []

        for individual_rate in rate:
            if individual_rate != 0:
                num_events, event_times, inter_arrival_times = generate_poisson_events(individual_rate, time_duration)
            else:
                num_events, event_times, inter_arrival_times = None, None, None
            num_events_list.append(num_events)
            event_times_list.append(event_times)
            inter_arrival_times_list.append(inter_arrival_times)

        return num_events_list, event_times_list, inter_arrival_times_list