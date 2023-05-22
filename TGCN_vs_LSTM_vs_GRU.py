import torch
import pickle

from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data, createLoaders
from src.models import TGCN, GRUModel, LSTMModel
from src.utils_training import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model(best_model, test_loader):
    import numpy as np
    
    # Load the best model and evaluate on the test set
    criterion = torch.nn.MSELoss()
    best_model.double()
    best_model.eval()
    best_model.to(device)

    # Evaluate the model on the test set
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, horizon_size, num_nodes = targets.size()
            final_output = torch.empty((batch_size, 0, num_nodes)).to(device)
            outputs = best_model(inputs.double())
            final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)
            for i in range(1, horizon_size):
                outputs = best_model(torch.cat((inputs[:, i:, :], final_output), dim=1))
                final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)
            loss = criterion(final_output, targets)
            test_loss += loss.item()
            # Save the predictions and actual values for plotting later
            predictions.append(final_output.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    test_loss /= len(test_loader)
    # print(f"Test Loss: {test_loss:.4f}")
    
    # Concatenate the predictions and actuals
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return (predictions, actuals)


# Not a parameter just the name of normalization use in preprocess_PeMS_data()
normalization = "center_and_reduce"

###############################################################################
# Parameters
###############################################################################
# Define the sliding window size, stride and horizon
_window_size = 84
horizon = 12
_stride = 1

# TGCN
num_epochs_TGCN = 200
# LSTM
num_epochs_LSTM_multivariate = 200
# GRU
num_epochs_GRU_multivariate = 200


df_PeMS_old, df_distance  = load_PeMS04_flow_data()

tgcn_dict = {}
lstm_dict = {}
gru_dict = {}


n_neighbors = 20
path_save_model = f"./{normalization}/nb_captor_{n_neighbors+1}/windows_{_window_size}_out_{horizon}"
df_PeMS, adjacency_matrix_PeMS, meanstd_dict = preprocess_PeMS_data(df_PeMS_old, df_distance, init_node=0, n_neighbors=n_neighbors, center_and_reduce=True)
model_path = f"{path_save_model}/epoch_{num_epochs_TGCN}/TGCN_model.pkl"

# # TGCN Model
model_TGCN = TGCN(adjacency_matrix_PeMS, hidden_dim=32, output_size=len(df_PeMS.columns))
train_loader_TGCN, val_loader_TGCN, test_loader_TGCN, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
tgcn_dict["testset"] = test_loader_TGCN
_ , _, _ = train_model(model_TGCN, train_loader_TGCN, val_loader_TGCN, model_path=model_path, num_epochs=num_epochs_TGCN, remove=False)

# # LSTM Model
model_multivariate_LSTM = LSTMModel(len(df_PeMS.columns), 32, len(df_PeMS.columns))
train_loader_LSTM, val_loader_LSTM, test_loader_LSTM, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
lstm_dict["testset"] = test_loader_LSTM
_ , _, _ = train_model(model_multivariate_LSTM, train_loader_LSTM, val_loader_LSTM,
                                            f"{path_save_model}/epoch_{num_epochs_LSTM_multivariate}/multivariate_LSTM_model.pkl", 
                                            num_epochs=num_epochs_LSTM_multivariate, remove=False)

# # GRU Model
model_multivariate_GRU = GRUModel(len(df_PeMS.columns), 32, len(df_PeMS.columns))
train_loader_GRU, val_loader_GRU, test_loader_GRU, _ = createLoaders(df_PeMS, window_size=_window_size, prediction_horizon=horizon)
gru_dict["testset"] = test_loader_GRU
_ , _, _ = train_model(model_multivariate_GRU, train_loader_GRU, val_loader_GRU, 
                                            f"{path_save_model}/epoch_{num_epochs_GRU_multivariate}/multivariate_GRU_model.pkl", 
                                            num_epochs=num_epochs_GRU_multivariate, remove=False)


######################################################################
# TGCN
######################################################################
# load best model
model_TGCN.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_TGCN}/TGCN_model.pkl".format(input)))

# Make predictions
predictions_TGCN, actuals_TGCN = test_model(model_TGCN, test_loader_TGCN)

nb_row, _, _ = predictions_TGCN.shape
predictions_TGCN = predictions_TGCN.reshape(nb_row * horizon, n_neighbors+1)
actuals_TGCN = actuals_TGCN.reshape(nb_row * horizon, n_neighbors+1)
tgcn_dict["true_value"] = actuals_TGCN
tgcn_dict["pred_value"] = predictions_TGCN 


######################################################################
# LSTM
######################################################################
# load best model
model_multivariate_LSTM.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_LSTM_multivariate}/multivariate_LSTM_model.pkl".format(input)))
    
# Make predictions
predictions_LSTM, actuals_LSTM = test_model(model_multivariate_LSTM, 
                                test_loader_LSTM)
predictions_LSTM = predictions_LSTM.reshape(nb_row * horizon, n_neighbors+1)
actuals_LSTM = actuals_LSTM.reshape(nb_row * horizon, n_neighbors+1)
lstm_dict["true_value"] = actuals_LSTM
lstm_dict["pred_value"] = predictions_LSTM 


######################################################################
# GRU
######################################################################
# Load best model
model_multivariate_GRU.load_state_dict(torch.load(f"{path_save_model}/epoch_{num_epochs_GRU_multivariate}/multivariate_GRU_model.pkl".format(input)))

# Make predictions
predictions_GRU, actuals_GRU = test_model(model_multivariate_GRU, 
                                test_loader_GRU)
predictions_GRU = predictions_GRU.reshape(nb_row * horizon, n_neighbors+1)
actuals_GRU = actuals_GRU.reshape(nb_row * horizon, n_neighbors+1)
gru_dict["true_value"] = actuals_GRU
gru_dict["pred_value"] = predictions_GRU

with open(f"{path_save_model}/tgcn_dict.pkl", 'wb') as file:
    pickle.dump(tgcn_dict, file)
with open(f"{path_save_model}/lstm_dict.pkl", 'wb') as file:
    pickle.dump(lstm_dict, file)
with open(f"{path_save_model}/gru_dict.pkl", 'wb') as file:
    pickle.dump(gru_dict, file)