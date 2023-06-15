###############################################################################
# Libraries
###############################################################################
from os import path


import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from metrics import rmse
from utils_streamlit_app import get_color_fed_vs_local, load_numpy, selection_of_experiment
from config import Params

st.set_page_config(layout="wide")


def plot_slider(experiment_path):
    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_sensor_with_nodes[sensor_select]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")

    def plot_box(title, ae, max_y_value, color):
        fig = go.Figure()
        box = go.Box(y=ae, marker_color=color, boxmean='sd', name=title, boxpoints="suspectedoutliers")
        fig.add_trace(box)
        fig.update_layout(
            title={
                'text': f"{title}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title=f"sensor_select {sensor_select}",
            yaxis_title="Trafic flow (absolute error)",
            yaxis=dict(range=[0, max_y_value]),
            font=dict(
                size=28,
                color="#FF7f7f"
            ),
            height=900, width=350
        )
        fig.update_traces(jitter=0)
        return fig

    ae_fed = (np.abs(y_pred_fed.flatten() - y_true.flatten()))
    ae_local = (np.abs(y_pred.flatten() - y_true.flatten()))
    max_y_value = max(max(ae_fed), max(ae_local))

    rmse_local = rmse(y_true.flatten(), y_pred.flatten())
    rmse_fed = rmse(y_true.flatten(), y_pred_fed.flatten())

    color_fed, color_local = get_color_fed_vs_local(rmse_fed, rmse_local, superior=False)

    # FEDERATED
    fed_fig = plot_box("Federated Prediction", ae_fed, max_y_value, color_fed)

    # LOCAL
    local_fig = plot_box("Alone Prediction", ae_local, max_y_value, color_local)
    with st.spinner('Plotting...'):
        st.subheader(f"Comparison between Federation and local version on sensor {sensor_select} (Absolute Error)")
        _, c2_fed_fig, c3_local_fig, _ = st.columns((1, 1, 1, 1))
        with c2_fed_fig:
            st.plotly_chart(fed_fig, use_container_width=False)
        with c3_local_fig:
            st.plotly_chart(local_fig, use_container_width=False)


#######################################################################
# Main
#######################################################################


st.header("Box Plot")

path_experiment_selected = selection_of_experiment()
if (path_experiment_selected is not None):

    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)

    mapping_sensor_with_nodes = {}
    for node in results.keys():
        mapping_sensor_with_nodes[config["nodes_to_filter"][int(node)]] = node

    if 'sensor_select' not in st.session_state:
        st.session_state['sensor_select'] = 0
    sensor_select = st.selectbox('Choose the sensor_select', mapping_sensor_with_nodes.keys(), index=st.session_state['sensor_select'])
    st.session_state['sensor_select'] = int(mapping_sensor_with_nodes[sensor_select])

    metrics = list(results[mapping_sensor_with_nodes[sensor_select]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, ["RMSE", "MAE", "SMAPE", "Superior Pred %"])

    federated_node = []
    if "Federated" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        federated_node = results[mapping_sensor_with_nodes[sensor_select]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["sensor in Federation"])

    local_node = []
    if "local_only" in results[mapping_sensor_with_nodes[sensor_select]].keys():
        local_node = results[mapping_sensor_with_nodes[sensor_select]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["sensor alone"])

    st.subheader("sensor in Federation vs sensor alone")
    fed_local_node = pd.concat((federated_node, local_node), axis=0)
    st.table(fed_local_node.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold'), ('color', 'black')]}]).format("{:.2f}"))

    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{path_experiment_selected}/y_true_local_{mapping_sensor_with_nodes[sensor_select]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_sensor_with_nodes[sensor_select]}.npy")):
        plot_slider(path_experiment_selected)
