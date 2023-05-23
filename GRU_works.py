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

# GRU
num_epochs_GRU_multivariate = 200

df_PeMS_old, df_distance  = load_PeMS04_flow_data()

tgcn_dict = {}
lstm_dict = {}
gru_dict = {}


n_neighbors = 19
path_save_model = f"/projets/fedvilagil/guy_experiments/nb_captor_{n_neighbors+1}/windows_{_window_size}_out_{horizon}"
df_PeMS, adjacency_matrix_PeMS, meanstd_dict = preprocess_PeMS_data(df_PeMS_old, df_distance, init_node=0, n_neighbors=n_neighbors, center_and_reduce=True)

# # GRU Model
model_multivariate_GRU = GRUModel(len(df_PeMS.columns), 32, len(df_PeMS.columns))
train_loader_GRU, val_loader_GRU, test_loader_GRU, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
gru_dict["testset"] = test_loader_GRU
_ , _, _ = train_model(model_multivariate_GRU, train_loader_GRU, val_loader_GRU, 
                                            model_path=f"{path_save_model}/epoch_{num_epochs_GRU_multivariate}/multivariate_GRU_model.pkl", 
                                            num_epochs=num_epochs_GRU_multivariate, remove=False)

######################################################################
# GRU
######################################################################
# Load best model
model_multivariate_GRU.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_GRU_multivariate}/multivariate_GRU_model.pkl".format(input)))

# Make predictions
predictions_GRU, actuals_GRU = testmodel(model_multivariate_GRU, test_loader_GRU, meanstd_dict=meanstd_dict, sensor_order_list=df_PeMS.columns)
gru_dict["true_value"] = actuals_GRU
gru_dict["pred_value"] = predictions_GRU

with open(f"{path_save_model}/gru_dict.pkl", 'wb') as file:
    pickle.dump(gru_dict, file)