import torch
import pickle

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, createLoaders
from src.models import TGCN, GRUModel, LSTMModel
from src.utils_training import train_model, testmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Not a parameter just the name of normalization use in preprocess_PeMS_data()
normalization = "center_and_reduce"

###############################################################################
# Parameters
###############################################################################
# Define the sliding window size, stride and horizon
_window_size = 42
horizon = 6
_stride = 1

# LSTM
num_epochs_LSTM_multivariate = 200

df_PeMS_old, df_distance  = load_PeMS04_flow_data()

tgcn_dict = {}
lstm_dict = {}
gru_dict = {}


n_neighbors = 19
path_save_model = f"/projets/fedvilagil/guy_experiments/nb_captor_{n_neighbors+1}/windows_{_window_size}_out_{horizon}"
df_PeMS, adjacency_matrix_PeMS, meanstd_dict = preprocess_PeMS_data(df_PeMS_old, df_distance, init_node=0, n_neighbors=n_neighbors, center_and_reduce=True)

# # LSTM Model
model_multivariate_LSTM = LSTMModel(len(df_PeMS.columns), 32, len(df_PeMS.columns))
train_loader_LSTM, val_loader_LSTM, test_loader_LSTM, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
lstm_dict["testset"] = test_loader_LSTM
_ , _, _ = train_model(model_multivariate_LSTM, train_loader_LSTM, val_loader_LSTM,
                                            model_path=f"{path_save_model}/epoch_{num_epochs_LSTM_multivariate}/multivariate_LSTM_model.pkl", 
                                            num_epochs=num_epochs_LSTM_multivariate, remove=False)

######################################################################
# LSTM
######################################################################
# load best model
model_multivariate_LSTM.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_LSTM_multivariate}/multivariate_LSTM_model.pkl".format(input)))

# Make predictions
predictions_LSTM, actuals_LSTM = testmodel(model_multivariate_LSTM, test_loader_LSTM, meanstd_dict=meanstd_dict, sensor_order_list=df_PeMS.columns)
lstm_dict["true_value"] = actuals_LSTM
lstm_dict["pred_value"] = predictions_LSTM 

with open(f"{path_save_model}/lstm_dict.pkl", 'wb') as file:
    pickle.dump(lstm_dict, file)