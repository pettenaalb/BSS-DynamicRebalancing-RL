import os
import pickle
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

action_bin_labels = ['STAY', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'DROP_BIKE', 'PICK_UP_BIKE', 'CHARGE_BIKE']

# Base paths for two training phases
BASE_PATH = ""
TRAINING_PATHS = {
    "Training Phase 1": os.path.join("training", "data"),
    "Training Phase 2": os.path.join("training_2", "data"),
}

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
                dcc.Graph(id="reward-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px'}),
                html.Label("Select Episode for Rewards:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-reward", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Failure plot section
            html.Div([
                html.Div([
                    html.Button("Update Failure Plot", id="update-btn-failure", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(id="failure-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px'}),
                html.Label("Select Episode for Failures:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-failure", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Q-value plot section
            html.Div([
                html.Div([
                    html.Button("Update Q-Value Plot", id="update-btn-q-value", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(id="q-value-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px'}),
                html.Label("Select Episode for Q-Values:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-q-value", value="all", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Action plot section
            html.Div([
                html.Div([
                    html.Button("Update Action Plot", id="update-btn-action", n_clicks=0,
                                style={'background-color': '#4A90E2', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer', 'border-radius': '5px'}),
                ], style={'margin-bottom': '10px'}),
                dcc.Graph(id="action-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px'}),
                html.Label("Select Episode for Actions:", style={'font-weight': 'bold'}),
                dcc.Dropdown(id="episode-selector-action", value="00", clearable=False, style={'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)', 'background-color': 'white'}),

            # Reward per Action, Epsilon and Loss plots
            html.Div([
                dcc.Graph(id="reward-action-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'}),
                dcc.Graph(id="epsilon-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px', 'margin-bottom': '20px'}),
                dcc.Graph(id="loss-plot", style={'border': '1px solid #d3d3d3', 'padding': '10px'})
            ], style={'width': '100%', 'padding': '10px', 'background-color': 'white', 'box-shadow': '0px 1px 3px rgba(0, 0, 0, 0.2)'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),
    ], style={'max-width': '1200px', 'margin': '0 auto'}),
], style={'font-family': 'Poppins, sans-serif', 'padding': '20px'})

def get_timeslot_options(training_path, default_option=True):
    if not os.path.exists(training_path):
        return []

    timeslot_folders = sorted(
        [folder for folder in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, folder))]
    )

    options = (
        [{"label": "All Episodes", "value": "all"}] if default_option else []
    ) + [{"label": folder, "value": folder} for folder in timeslot_folders]
    return options

@app.callback(
    [Output("episode-selector-reward", "options"),
     Output("episode-selector-failure", "options"),
     Output("episode-selector-q-value", "options"),
     Output("episode-selector-action", "options")],
    [Input("training-selector", "value"),
        Input("interval-component", "n_intervals")]
)
def update_timeslot_dropdown(training_path, n_intervals):
    options = get_timeslot_options(training_path)
    return [options] * 4

def load_results(training_path, timeslot_folder="all"):
    if timeslot_folder == "all":
        rewards, failures, q_values, loss, epsilon, actions, rewards_per_action = [], [], [], [], [], [], []
        timeslot_folders = get_timeslot_options(training_path)[1:]  # Exclude "All Timeslots" option
        for folder in [opt['value'] for opt in timeslot_folders]:
            r, f, q, l, e, a, ra = load_results(training_path, folder)
            rewards.extend(r)
            failures.extend(f)
            q_values.extend(q)
            loss.extend(l)
            epsilon.extend(e)
            rewards_per_action.extend(ra)
            actions = [sum(x) for x in zip(actions, a)]
        return rewards, failures, q_values, loss, epsilon, actions, rewards_per_action

    timeslot_path = os.path.join(training_path, timeslot_folder)
    if not os.path.exists(timeslot_path):
        return [], [], [], [], [], [], []

    with open(os.path.join(timeslot_path, 'rewards_per_timeslot.pkl'), 'rb') as f:
        rewards_per_timeslot = pickle.load(f)
    with open(os.path.join(timeslot_path, 'failures_per_timeslot.pkl'), 'rb') as f:
        failures_per_timeslot = pickle.load(f)
    with open(os.path.join(timeslot_path, 'q_values_per_timeslot.pkl'), 'rb') as f:
        q_values_per_timeslot = pickle.load(f)
    with open(os.path.join(timeslot_path, 'losses.pkl'), 'rb') as f:
        loss_per_timeslot = pickle.load(f)
    with open(os.path.join(timeslot_path, 'reward_per_action.pkl'), 'rb') as f:
        rewards_per_action = pickle.load(f)

    rewards = [r for r, _ in rewards_per_timeslot]
    failures = [f for f, _ in failures_per_timeslot]
    q_values = [np.mean(q) for q, _ in q_values_per_timeslot]
    epsilon = [e for _, e in rewards_per_timeslot]

    with open(os.path.join(timeslot_path, 'action_per_step.pkl'), 'rb') as f:
        action_per_step = pickle.load(f)

    action_bins = [0] * len(action_bin_labels)
    for action, _ in action_per_step:
        action_bins[action] += 1

    return rewards, failures, q_values, loss_per_timeslot, epsilon, action_bins, rewards_per_action

def create_plot(data, title, y_axis_label, x_axis_label, cumulative=False, action_plot=False, double_x_axis=False):
    if not data:
        return go.Figure().update_layout(title=title, yaxis_title=y_axis_label)

    if action_plot:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=action_bin_labels, y=data))
        fig.update_layout(
            title=title,
            yaxis_title=y_axis_label,
            legend=dict(
                x=0.9,  # Horizontal position (0 = left, 1 = right)
                y=0.9,  # Vertical position (0 = bottom, 1 = top)
                bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                bordercolor='black',
                borderwidth=1
            )
        )
    else:
        y_cumulative = np.cumsum(data) / np.arange(1, len(data) + 1) if cumulative else []
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data, mode='lines', name="Values"))
        if cumulative:
            fig.add_trace(go.Scatter(y=y_cumulative, mode='lines', name="Cumulative Mean", line=dict(color='red')))

        fig.update_layout(
            title=title,
            yaxis_title=y_axis_label,
            xaxis_title=x_axis_label,
            legend=dict(
                x=0.84,  # Horizontal position (0 = left, 1 = right)
                y=0.97,  # Vertical position (0 = bottom, 1 = top)
                bgcolor='rgba(255, 255, 255, 1)',
                bordercolor='black',
                borderwidth=1
            )
        )
    return fig

@app.callback(
    Output("reward-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-reward", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-reward", "value")]
)
def update_reward_plot(n_intervals, n_clicks, training_path, timeslot_folder):
    rewards, *_ = load_results(training_path, timeslot_folder)
    return create_plot(rewards, "Rewards per Timeslot", "Reward", "Timeslot", cumulative=True, double_x_axis=True)

@app.callback(
    Output("failure-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-failure", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-failure", "value")]
)
def update_failure_plot(n_intervals, n_clicks, training_path, timeslot_folder):
    _, failures, *_ = load_results(training_path, timeslot_folder)
    return create_plot(failures, "Failures per Timeslot", "Failures", "Timeslot", cumulative=True, double_x_axis=True)

@app.callback(
    Output("q-value-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-q-value", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-q-value", "value")]
)
def update_q_value_plot(n_intervals, n_clicks, training_path, timeslot_folder):
    _, _, q_values, *_ = load_results(training_path, timeslot_folder)
    return create_plot(q_values, "Q-Values per Timeslot", "Q-Value", "Timeslot", cumulative=True, double_x_axis=True)

@app.callback(
    [Output("epsilon-plot", "figure"),
     Output("loss-plot", "figure"),
     Output("reward-action-plot", "figure")],
    [Input("interval-component", "n_intervals"),
     Input("training-selector", "value")]
)
def update_epsilon_loss(n_intervals, training_path):
    *_, loss, epsilon, _, rewards_per_action = load_results(training_path, "all")
    epsilon_plot = create_plot(epsilon, "Epsilon over Time", "Epsilon", "Timeslot", cumulative=False)
    loss_plot = create_plot(loss, "Loss over Time", "Loss", "Timeslot", cumulative=False)
    reward_action_plot = create_plot(rewards_per_action, "Reward per Action", "Reward", "Timeslot", action_plot=True)
    return epsilon_plot, loss_plot, reward_action_plot

@app.callback(
    Output("action-plot", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("update-btn-action", "n_clicks"),
     Input("training-selector", "value"),
     Input("episode-selector-action", "value")]
)

def update_action_plot(n_intervals, n_clicks, training_path, timeslot_folder):
    *_, actions, _ = load_results(training_path, timeslot_folder)
    return create_plot(actions, "Actions per Timeslot", "Actions", "Timeslot", action_plot=True)


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)