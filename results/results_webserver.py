import os
import pickle
import numpy as np
import dash
import osmnx as ox
from dash import dcc, html
from dash.dependencies import Input, Output
from utils import load_results, get_episode_options, create_plot, action_bin_labels, generate_osmnx_graph, initialize_graph

# insert here the training runs to evaluate
phases = [17,26,27]
port = 8050

# Base paths for two training phases
BASE_PATH = "results/"
TRAINING_PATHS = {}
for n in phases :
    label = f"Phase {n} Training"
    folder = f"training_{n}"
    TRAINING_PATHS.update({label: os.path.join(BASE_PATH, folder, "data")})
# for n in phases :
#     label = f"Phase {n} Validation"
#     folder = f"validation_{n}"
#     TRAINING_PATHS.update({label: os.path.join(BASE_PATH, folder, "data")})

# "Training Phase 3": os.path.join(BASE_PATH, "training_3", "data"),
# "Training Phase 3": os.path.join(BASE_PATH, "training_3", "data"),
# "Training Phase 4": os.path.join(BASE_PATH, "training_4", "data"),
# "Validation Phase 2": os.path.join(BASE_PATH, "validation_2", "data"),
# "Validation Phase 3": os.path.join(BASE_PATH, "validation_3", "data"),
# "Validation Phase 4": os.path.join(BASE_PATH, "validation_4", "data"),


# Import external stylesheets (Google Fonts)
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap']

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Layout with dropdown selector, auto-update, and plots
app.layout = html.Div([
    html.H1("Training Results Dashboard", style={'text-align': 'center', 'color': '#4A90E2'}),

    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # 5 minutes in milliseconds
        n_intervals=0
    ),

    html.Div([
        # Dropdown for training phase selection
        html.Div([
            html.Label("Select Training Phase:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id="training-selector",
                options=[{"label": name, "value": path} for name, path in TRAINING_PATHS.items()],
                value=list(TRAINING_PATHS.values())[0],  # Default to first training phase
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'margin-bottom': '20px'}),

        html.Div([
            # Reward plot section
            html.Div([
                html.Div([
                    html.Button("Update Reward Plot", id="update-btn-reward", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'rewards_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Rewards:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-reward", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Reward tracking plot section
            html.Div([
                html.Div([
                    html.Button("Update Reward Tracking Plot", id="update-btn-reward-tracking", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-tracking-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'reward_tracking_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Action:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="action-selector-reward-tracking",
                    options=[{'label': label, 'value': label} for label in action_bin_labels],
                    value=action_bin_labels[0],
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Failure plot section
            html.Div([
                html.Div([
                    html.Button("Update Failure Plot", id="update-btn-failure", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="failure-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'failures_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Failures:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-failure", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Q-value plot section
            html.Div([
                html.Div([
                    html.Button("Update Q-Value Plot", id="update-btn-q-value", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="q-value-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'q_values_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Q-Values:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-q-value", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Action plot section
            html.Div([
                html.Div([
                    html.Button("Update Action Plot", id="update-btn-action", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="action-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'actions_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                html.Label("Select Episode for Actions:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-action", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Reward per Action, Epsilon and Loss plots
            html.Div([
                html.Div([
                    html.Button("Update Plots", id="update-btn-plots", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(
                    id="reward-action-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'reward_per_action_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="epsilon-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'epsilon_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="loss-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'loss_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="failures-baseline-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'failures_baseline_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    }),
                dcc.Graph(
                    id="deployed-bikes-plot",
                    style={'border': '1px solid #d3d3d3', 'padding': '10px'},
                    config={
                        'toImageButtonOptions': {
                            'format': 'jpeg',  # Available formats: 'svg', 'png', 'jpeg', 'webp'
                            'filename': 'deployed_bikes_plot',
                            'height': 800,   # Image height in pixels
                            'width': 1200,   # Image width in pixels
                            'scale': 3       # Multiply resolution (e.g., for high-DPI)
                        }
                    })
            ], style={'width': '100%', 'padding': '10px', 'background-color': 'white', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)'}),
            html.Div([
                html.Label("Select Episode for Rewards:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="metric-selector-graph",
                    options=[{"label": "Visits", "value": "visits"},
                             {"label": "Critic Score", "value": "critic_score"},
                             {"label": "Num of bikes", "value": "num_bikes"}],
                    value="visits",
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Button("Update OSMnx Graph", id="update-btn-osmnx", n_clicks=0,
                            style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none',
                                   'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                html.Img(id="osmnx-graph", style={'width': '70%', 'border': '1px solid #d3d3d3', 'padding': '10px'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),
    ], style={'max-width': '1200px', 'margin': '0 auto'}),
], style={'font-family': 'Poppins, sans-serif', 'padding': '20px'})

@app.callback(
    [Output("episode-selector-reward", "options"),
     Output("episode-selector-failure", "options"),
     Output("episode-selector-q-value", "options"),
     Output("episode-selector-action", "options")],
    [Input("training-selector", "value"),
        Input("interval-component", "n_intervals")]
)
def update_episode_dropdown(training_path, n_intervals):
    options = get_episode_options(training_path)
    first_option = options[:1]
    reversed_options = first_option + options[1:][::-1]
    return [reversed_options] * 4

@app.callback(
    Output("reward-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-reward", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-reward", "value")]
)
def update_reward_plot(n_intervals, n_clicks, training_path, episode_folder):
    rewards = load_results(training_path, episode_folder, metric="rewards_per_timeslot")
    return create_plot(rewards, "Rewards per Timeslot", "Reward", "Timeslot", cumulative=True)

@app.callback(
    Output("failure-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-failure", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-failure", "value")]
)
def update_failure_plot(n_intervals, n_clicks, training_path, episode_folder):
    failures = load_results(training_path, episode_folder, metric="failures_per_timeslot")
    return create_plot(failures, "Failures per Timeslot", "Failures", "Timeslot",
                       cumulative=True, failures_plot=True)

@app.callback(
    Output("q-value-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-q-value", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-q-value", "value")]
)
def update_q_value_plot(n_intervals, n_clicks, training_path, episode_folder):
    q_values_per_timeslot = load_results(training_path, episode_folder, metric="q_values_per_timeslot")

    q_values = []
    for q_value in q_values_per_timeslot:
        q_values.append(np.mean(q_value))

    return create_plot(q_values, "Q-Values per Timeslot", "Q-Value", "Timeslot", cumulative=True)

@app.callback(
    [Output("epsilon-plot", "figure"),
     Output("loss-plot", "figure"),
     Output("reward-action-plot", "figure"),
     Output("deployed-bikes-plot", "figure")],
    [Input("interval-component", "n_intervals"),
     Input("training-selector", "value"),
     Input("update-btn-plots", "n_clicks")]
)
def update_epsilon_loss(n_intervals, training_path, n_clicks):
    # loss = load_results(training_path, metric="losses")
    try:
        epsilon = load_results(training_path, metric="epsilon_per_timeslot")
        epsilon_plot = create_plot(epsilon, "Epsilon over Time", "Epsilon", "Timeslot", cumulative=False)
    except FileNotFoundError:
        epsilon_plot = None
    deployed_bikes = load_results(training_path, metric="deployed_bikes")
    last_episode_folder = get_episode_options(training_path)[-1]
    last_episode_reward_tracking = load_results(training_path, last_episode_folder['value'], metric="reward_tracking")

    actns_len = len(last_episode_reward_tracking)
    reward_per_action = [0] * actns_len
    for action, _ in enumerate(action_bin_labels[:actns_len]):
        reward_per_action[action] = np.mean(last_episode_reward_tracking[action])

    loss_plot = create_plot([], "Loss over Time", "Loss", "Timeslot", cumulative=False)
    reward_action_plot = create_plot(reward_per_action, "Reward per Action", "Reward", "Timeslot", action_plot=True)
    deployed_bikes_plot = create_plot(deployed_bikes, "Deployed Bikes per Timeslot", "Bikes", "Timeslot", cumulative=False)

    return epsilon_plot, loss_plot, reward_action_plot, deployed_bikes_plot

@app.callback(
    Output("action-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-action", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-action", "value")]
)

def update_action_plot(n_intervals, n_clicks, training_path, episode_folder):
    action_per_step = load_results(training_path, episode_folder, metric="action_per_step")

    action_bins = [0] * len(action_bin_labels)
    for action in action_per_step:
        action_bins[action] += 1

    fig = create_plot(action_bins, "Actions per Timeslot", "Actions", "Timeslot", action_plot=True)

    # Add an annotation to display the selected timeslot
    fig.add_annotation(
        x=0.99,  # Near the right edge
        y=0.98,  # Slightly above the plot area
        xref="paper",  # Relative to the whole plot area
        yref="paper",
        text=f"Episode: {episode_folder}",  # Display the current timeslot
        showarrow=False,
        font=dict(
            family="Poppins, sans-serif",
            size=12,
            color="black"
        ),
        align="right",
        bgcolor="rgba(255, 255, 255, 1.0)",  # Optional background for better readability
        bordercolor="black",
        borderwidth=1
    )

    return fig

@app.callback(
    Output("failures-baseline-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-plots", "n_clicks")]
)

def update_failures_baseline_plot(interval, n_clicks):
    with open(os.path.join('benchmarks/results/total_failures.pkl'), 'rb') as f:
        failures = pickle.load(f)
    return create_plot(failures, "Failures", "Failures", "Timeslot", cumulative=True,
                       failures_plot=True)

@app.callback(
    Output("reward-tracking-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-reward-tracking", "n_clicks"),
     Input("training-selector", "value"),
     Input("action-selector-reward-tracking", "value")]
)
def update_reward_tracking_plot(n_intervals, n_clicks, training_path, action):
    reward_tracking = load_results(training_path, "all", metric="reward_tracking")
    action_value = action_bin_labels.index(action)
    rewards = reward_tracking[action_value]
    return create_plot(rewards, "Rewards per Step", "Reward", "Step", cumulative=True)


@app.callback(
    Output("osmnx-graph", "src"),
    [Input("training-selector", "value"),
     Input("metric-selector-graph", "value"),
     Input("update-btn-osmnx", "n_clicks")]
)
def update_osmnx_graph(training_path, metric, n_clicks):
    graph = initialize_graph('data/utils/cambridge_network.graphml')
    # Initialize the cells
    with open('data/utils/cell_data.pkl', 'rb') as file:
        cells = pickle.load(file)

    # if training_path == "training_2/data" or training_path == "validation_2/data":
    #     graph = initialize_graph('../data_cambridge_full/utils/cambridge_network.graphml')
    #     # Initialize the cells
    #     with open('../data_cambridge_full/utils/cell_data.pkl', 'rb') as file:
    #         cells = pickle.load(file)

    last_episode_folder = get_episode_options(training_path)[-1]
    cell_subgraph = load_results(training_path, last_episode_folder['value'], metric="cell_subgraph")
    nodes = ox.graph_to_gdfs(cell_subgraph, nodes=True, edges=False)

    cell_values = {}
    total_visits = 0
    for _, node in nodes.iterrows():
        cell_values[node['cell_id']] = node[metric]
        total_visits += node['visits']

    if metric == "visits":
        for cell_id in cell_values.keys():
            cell_values[cell_id] = cell_values[cell_id] / total_visits

    return generate_osmnx_graph(graph, cells, cell_values, percentage=(metric == "visits" or metric == "critic_score"))


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)