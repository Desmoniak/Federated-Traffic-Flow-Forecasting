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

# TGCN
num_epochs_TGCN = 200

df_PeMS_old, df_distance  = load_PeMS04_flow_data()

tgcn_dict = {}
lstm_dict = {}
gru_dict = {}


n_neighbors = 19
path_save_model = f"/projets/fedvilagil/guy_experiments/{normalization}/nb_captor_{n_neighbors+1}/windows_{_window_size}_out_{horizon}"
df_PeMS, adjacency_matrix_PeMS, meanstd_dict = preprocess_PeMS_data(df_PeMS_old, df_distance, init_node=0, n_neighbors=n_neighbors, center_and_reduce=True)

# # TGCN Model
model_TGCN = TGCN(adjacency_matrix_PeMS, hidden_dim=32, output_size=len(df_PeMS.columns))
train_loader_TGCN, val_loader_TGCN, test_loader_TGCN, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
tgcn_dict["testset"] = test_loader_TGCN
_ , _, _ = train_model(model_TGCN, train_loader_TGCN, val_loader_TGCN, 
                                            model_path=f"{path_save_model}/epoch_{num_epochs_TGCN}/TGCN_model.pkl",
                                            num_epochs=num_epochs_TGCN, remove=False)

######################################################################
# TGCN
######################################################################
# load best model
model_TGCN.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_TGCN}/TGCN_model.pkl".format(input)))

# Make predictions
predictions_TGCN, actuals_TGCN = testmodel(model_TGCN, test_loader_TGCN, meanstd_dict=meanstd_dict, sensor_order_list=df_PeMS.columns)
tgcn_dict["true_value"] = actuals_TGCN
tgcn_dict["pred_value"] = predictions_TGCN 

with open(f"{path_save_model}/tgcn_dict.pkl", 'wb') as file:
    pickle.dump(tgcn_dict, file)
