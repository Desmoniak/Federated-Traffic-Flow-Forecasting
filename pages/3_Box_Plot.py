###############################################################################
# Libraries
###############################################################################
from os import path

import glob
import json
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from src.metrics import rmse
from src.config import Params


@st.cache_data
def filtering_path_file(file_dict, filter_list):
    """
    Returns a new dictionary that contains only the files that are in the filter list.

    Parameters
    ----------
    file_dict : dict
        The dictionary that maps some keys to lists of files.
    filter_list : list
        The list of files that are used as a filter.

    Return
    ------
    filtered_file_dict : dict
        The new dictionary that contains only the keys and files from file_dict that are also in filter_list.
    """
    filtered_file_dict = {}
    for key in file_dict.keys():
        for file in file_dict[key]:
            if file in filter_list:
                if key in filtered_file_dict:
                    filtered_file_dict[key].append(file)
                else:
                    filtered_file_dict[key] = [file]
    return filtered_file_dict

@st.cache_data
def load_numpy(path):
    return np.load(path)


def plot_slider(experiment_path):
    test_set = load_numpy(f"{params.save_model_path}/test_data_{mapping_captor_with_nodes[captor]}.npy")

    y_true = load_numpy(f"{experiment_path}/y_true_local_{mapping_captor_with_nodes[captor]}.npy")
    y_pred = load_numpy(f"{experiment_path}/y_pred_local_{mapping_captor_with_nodes[captor]}.npy")
    y_pred_fed = load_numpy(f"{experiment_path}/y_pred_fed_{mapping_captor_with_nodes[captor]}.npy")
    
    index = load_numpy(f"{params.save_model_path}/index_{mapping_captor_with_nodes[captor]}.npy")
    index = pd.to_datetime(index, format='%Y-%m-%dT%H:%M:%S.%f')
    
    def plot_box_plot(color, label, title, y_pred, y_true):
        fig = px.box(y=(np.abs(y_pred.flatten() - y_true.flatten())), color_discrete_sequence=['grey'], title=title, points="suspectedoutliers")
        fig.update_layout(
        title={
            'text': f"{title} Absolute error",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f"captor {captor}",
        yaxis_title="Trafic flow (absolute error)",
        font=dict(
            size=28,
            color="#7f7f7f"
        ),
        width=500,
        height=800
        )
        return fig

    # FEDERATED
    fed_fig = plot_box_plot('green', 'Federated Prediction', "Federated Prediction", y_pred_fed, y_true)
    
    # LOCAL
    local_fig = plot_box_plot('red', 'Local Prediction', "Local Prediction", y_pred, y_true)
    
    with st.spinner('Plotting...'):
        st.plotly_chart(fed_fig, use_container_width=True)
        st.plotly_chart(local_fig, use_container_width=True)



@st.cache_data
def map_path_experiments_to_params(path_files, params_config_use_for_select):
    """
    Map all path experiments with parameters of their config.json

    Parameters:
    -----------
        path_files : 
            The path to the config.json of all experiments
        params_config_use_for_select :
            The parameters use for the selection
            
    Returns:
        The mapping between path to the experiments and parameters of the experiements
        exemple :
        mapping = {
            "nb_node" : {
                3: [path_1, path_3],
                4: [path_4]
            },
            windows_size: {
                1: [path_1],
                5: [path_3],
                8: [path_4]
            }
        }
    """
    mapping_path_with_param = {}
    for param in params_config_use_for_select:
        mapping_path_with_param[param] = {}
        for file in path_files:
            with open(file) as f:
                config = json.load(f)
            if config[param] in mapping_path_with_param[param].keys():
                mapping_path_with_param[param][config[param]].append(file)
            else:
                mapping_path_with_param[param][config[param]] = [file]
    return mapping_path_with_param

def selection_of_experiment(possible_choice):
    nb_captor = st.selectbox('Choose the number of captor', possible_choice["number_of_nodes"].keys())

    windows_size_filtered = filtering_path_file(possible_choice["window_size"], possible_choice["number_of_nodes"][nb_captor])
    window_size = st.selectbox('Choose the windows size', windows_size_filtered.keys())
        
    horizon_filtered = filtering_path_file(possible_choice["prediction_horizon"], windows_size_filtered[window_size])
    horizon_size = st.selectbox('Choose the prediction horizon', horizon_filtered.keys())
    
    models_filtered = filtering_path_file(possible_choice["model"], horizon_filtered[horizon_size])
    model = st.selectbox('Choose the model', models_filtered.keys())
    
    if(len(models_filtered[model]) > 1):
        st.write("TODO : WARNING ! More than one results correspond to your research pick only one (see below)")
        select_exp = st.selectbox("Choose", models_filtered[model])
        select_exp = models_filtered[model].index(select_exp)
        experiment_path = ("\\".join(models_filtered[model][select_exp].split("\\")[:-1]))
        
    else:
        experiment_path = ("\\".join(models_filtered[model][0].split("\\")[:-1]))
    
    return experiment_path



#######################################################################
# Main
#######################################################################
experiments = "experiments" # PATH where your experiments are saved
if path_files := glob.glob(f"./{experiments}/**/config.json", recursive=True):
    
    params_config_use_for_select = \
    [
        "number_of_nodes",
        "window_size",
        "prediction_horizon",
        "model"
    ]
    user_selection = map_path_experiments_to_params(path_files, params_config_use_for_select)

    path_experiment_selected = selection_of_experiment(user_selection)

    with open(f"{path_experiment_selected}/test.json") as f:
        results = json.load(f)
    with open(f"{path_experiment_selected}/config.json") as f:
        config = json.load(f)


    mapping_captor_with_nodes = {}
    for node in results.keys():
        mapping_captor_with_nodes[config["nodes_to_filter"][int(node)]] = node


    captor = st.selectbox('Choose the captor', mapping_captor_with_nodes.keys())


    metrics = list(results[mapping_captor_with_nodes[captor]]["local_only"].keys())
    multiselect_metrics = st.multiselect('Choose your metric(s)', metrics, metrics)


    local_node = []
    if "local_only" in results[mapping_captor_with_nodes[captor]].keys():
        local_node = results[mapping_captor_with_nodes[captor]]["local_only"]
        local_node = pd.DataFrame(local_node, columns=multiselect_metrics, index=["Captor alone"])

    federated_node = []
    if "Federated" in results[mapping_captor_with_nodes[captor]].keys():
        federated_node = results[mapping_captor_with_nodes[captor]]["Federated"]
        federated_node = pd.DataFrame(federated_node, columns=multiselect_metrics, index=["Captor in Federation"])

    st.dataframe(pd.concat((federated_node, local_node), axis=0), use_container_width=True)


    params = Params(f'{path_experiment_selected}/config.json')
    if (path.exists(f'{params.save_model_path}y_true_local_{mapping_captor_with_nodes[captor]}.npy') and
        path.exists(f"{path_experiment_selected}/y_pred_fed_{mapping_captor_with_nodes[captor]}.npy")):
        plot_slider(path_experiment_selected)