from os import makedirs
import sys


import torch
import importlib
import contextlib
import numpy
import json
from pathlib import Path


from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, local_dataset
from src.utils_training import testmodel
from src.metrics import calculate_metrics, metrics_table, Percentage_of_Superior_Predictions
import src.config

seed = 42
torch.manual_seed(seed)


# Get the path to the configuration file from the command-line arguments
if len(sys.argv) != 2:
    print("Usage: python3 experiment.py CONFIG_FILE_PATH")
    sys.exit(1)
config_file_path = sys.argv[1]

params = src.config.Params(config_file_path)

PATH_EXPERIMENTS = Path("experiments") / params.save_model_path

makedirs(PATH_EXPERIMENTS, exist_ok=True)

with open(PATH_EXPERIMENTS / 'test.txt', 'w') as f:
    with contextlib.redirect_stdout(src.config.Tee(f, sys.stdout)):

        class_name = params.model
        module = importlib.import_module('src.models')
        model = getattr(module, class_name)

        input_size = 1
        hidden_size = 32
        num_layers = 6
        output_size = 1

        #  Load traffic flow dataframe and graph dataframe from PEMS
        df_PeMS, distance = load_PeMS04_flow_data()
        df_PeMS, adjmat, meanstd_dict = preprocess_PeMS_data(df_PeMS, params.time_serie_percentage_length,
                                                            distance, params.init_node, params.n_neighbours,
                                                            params.smooth, params.center_and_reduce,
                                                            params.normalize, params.sort_by_mean)
        if params.nodes_to_filter == []:
            params.nodes_to_filter = list(df_PeMS.columns[:params.number_of_nodes])
            with open(PATH_EXPERIMENTS / "config.json", 'r') as file:
                data = json.load(file)
                data["nodes_to_filter"] = params.nodes_to_filter
                with open(PATH_EXPERIMENTS / "config.json", 'w') as file:
                    json.dump(data, file, indent=4, separators=(',', ': '))
        print(params.nodes_to_filter)
        datadict = local_dataset(df=df_PeMS,
                                nodes=params.nodes_to_filter,
                                window_size=params.window_size,
                                stride=params.stride,
                                prediction_horizon=params.prediction_horizon,
                                batch_size=params.batch_size)
        print(datadict.keys())
        metrics_dict = {}

        for node in range(len(params.nodes_to_filter)):
            metrics_dict[node] = {}
            datadict[node]['test_data'] = datadict[node]['test_data'] * meanstd_dict[params.nodes_to_filter[node]]['std'] + meanstd_dict[params.nodes_to_filter[node]]['mean']

            numpy.save(PATH_EXPERIMENTS / f"test_data_{node}", datadict[node]['test_data'])
            numpy.save(PATH_EXPERIMENTS / f"index_{node}", datadict[node]['test_data'].index)

            y_true, y_pred = testmodel(model(params.model_input_size,
                                            params.model_hidden_size,
                                            params.model_output_size,
                                            params.model_num_layers),
                                            datadict[node]['test'],
                                            PATH_EXPERIMENTS / f'local{node}.pth',
                                            meanstd_dict=meanstd_dict,
                                            sensor_order_list=[params.nodes_to_filter[node]])
            local_metrics = calculate_metrics(y_true, y_pred)
            metrics_dict[node]['local_only'] = local_metrics
            numpy.save(PATH_EXPERIMENTS / f'y_true_local_{node}', y_true)
            numpy.save(PATH_EXPERIMENTS / f'y_pred_local_{node}', y_pred)

            y_true_fed, y_pred_fed = testmodel(model(params.model_input_size,
                                                    params.model_hidden_size,
                                                    params.model_output_size,
                                                    params.model_num_layers),
                                                    datadict[node]['test'],
                                                    PATH_EXPERIMENTS / f'bestmodel_node{node}.pth',
                                                    meanstd_dict=meanstd_dict,
                                                    sensor_order_list=[params.nodes_to_filter[node]])

            fed_metrics = calculate_metrics(y_true_fed, y_pred_fed)
            metrics_dict[node]['Federated'] = fed_metrics
            numpy.save(PATH_EXPERIMENTS / f'y_true_fed_{node}', y_true_fed)
            numpy.save(PATH_EXPERIMENTS / f'y_pred_fed_{node}', y_pred_fed)
            print(f'Federated vs local only for node {node} :')
            fed_metrics['Superior Pred %'], local_metrics['Superior Pred %'] = Percentage_of_Superior_Predictions(y_true, y_pred, y_true_fed, y_pred_fed)
            print(metrics_table({'Local': local_metrics, 'Federated': fed_metrics}))

with open(PATH_EXPERIMENTS / "test.json", "w") as outfile:
    json.dump(metrics_dict, outfile)
