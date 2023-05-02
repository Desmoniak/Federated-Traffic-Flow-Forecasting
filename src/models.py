"""
@CREDITS
###############################################################################
The TGCN model is based on :
    The article "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction"
    IEEE Transactions on Intelligent Transportation Systems
    DOI = 10.1109/TITS.2019.2935152
    year = 2019
###############################################################################
"""



import torch
import copy
from src.utils_graph import compute_laplacian_with_self_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(torch.nn.Module):
    """
    Class to define LSTM model here with 6 LSTM layers and 1 fully connected layer by default
    Parameters

    ----------
    input_size : int

    hidden_size : int
        number of hidden unit.

    output_size : int

    num_layer : int = 6
        number of layer.
    """
    def __init__(self, input_size : int, hidden_size : int, output_size : int, num_layers : int=6):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(torch.nn.Module):
    
    """
    Class to define GRU model here with 6 GRU layers and 1 fully connected layer by default

    Parameters
    ----------
    input_size : int

    hidden_size : int
        number of hidden unit.

    output_size : int

    num_layer : int = 6
        number of layer.
    """
    
    def __init__(self, input_size : int, hidden_size : int, output_size : int, num_layers : int=6):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class TGCNGraphConvolution(torch.nn.Module):
    
    """
    Class to define TGCNGraphConvolution that is use by class TGCNCell

    Parameters
    ----------
    adj : matrix
        adjacency matrix

    num_gru_units : int
        number of hidden unit.

    output_size : int

    bias : int = 0.0
        default bias.
    """

    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", compute_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = torch.nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(torch.nn.Module):
    
    """
    Class to define TGCNCell that is use by class TGCN

    Parameters
    ----------
    adj : matrix
        adjacency matrix

    input_dim : int

    hidden_dim : int
        number of hidden unit.

    num_layer : int = 1
        number of layer.
    """

    def __init__(self, adj, input_dim: int, hidden_dim: int, num_layer : int = 1):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layer = num_layer
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):

        for _ in range(self._num_layer):
            # [r, u] = sigmoid(A[x, h]W + b)
            # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
            concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
            # r (batch_size, num_nodes, num_gru_units)
            # u (batch_size, num_nodes, num_gru_units)
            r, u = torch.chunk(concatenation, chunks=2, dim=1)
            # c = tanh(A[x, (r * h)W + b])
            # c (batch_size, num_nodes * num_gru_units)
            c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
            # h := u * h + (1 - u) * c
            # h (batch_size, num_nodes * num_gru_units)
            hidden_state = u * hidden_state + (1.0 - u) * c
        new_hidden_state = hidden_state
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim,
                "hidden_dim": self._hidden_dim,
                "num_layer": self._num_layer
        }


class TGCN(torch.nn.Module):
    
    """
    Class to define TGCN

    Parameters
    ----------
    adj : matrix
        adjacency matrix

    hidden_dim : int
        number of hidden unit.

    output_size : int = 1

    num_layer : int = 1
        number of layer.
    """
    
    def __init__(self, adjacency_matrix, hidden_dim: int=64, output_size: int=1, num_layer : int= 1):
        super(TGCN, self).__init__()
        self._input_dim = adjacency_matrix.shape[0]
        self._hidden_dim = hidden_dim
        self._output_size = output_size
        self._num_layer = num_layer
        self.register_buffer("adj", torch.FloatTensor(adjacency_matrix))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim, self._num_layer)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        batch_size, windows_size, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(windows_size):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
        # ouput (batch_size, num_node, self.hidden_dim)
        output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        # ouput (batch_size, num_node, self.hidden_dim) => (batch_size, num_node, 1)
        output = self.fc(output)
        # ouput (batch_size, num_node, 1) => (batch_size, num_node)
        output = output.reshape(batch_size, num_nodes)
        return output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim,
                "hidden_dim": self._hidden_dim,
                "output_size": self._output_size,
                "num_layer": self._num_layer
        }


import os
import matplotlib.pyplot as plt
def train_model(model, train_loader, val_loader, model_path, num_epochs = 200, remove = False):
    """
    Train a model

    Parameters
    ----------
    model : any
        model to train.

    train_loader : DataLoader
        Train Dataloader.

    val_loader : Dataloader
        Valid Dataloader.

    model_path : string
        Where to save the model after training it.

    num_epoch : int=200
        How many epochs to train the model.

    remove : bool=False
        Remove the model after the training phase.
    """

    # Train your model and evaluate on the validation set
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.squeeze().to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        val_loss = 0.0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.squeeze().to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            val_loss += loss.item()            
        val_loss /= len(val_loader)
        valid_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path_folder = model_path.split("/")[:-1]
            new_path = '/'.join(path_folder)
            os.makedirs(new_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        # print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    best_model =  copy.deepcopy(model)
    best_model.load_state_dict(torch.load(model_path))
    if remove:
        os.remove(model_path)
    
    return best_model, valid_losses, train_losses


def testmodel(best_model,test_loader, path='local.pth', plot =False, criterion = torch.nn.MSELoss(), percentage_error_fix = 1):
    from src.metrics import rmse, rmspe, maape, mape 
    from sklearn.metrics import mean_absolute_error

    """
    Test model using test data

    Parameters
    ----------
    best_model : any
        model to test.

    test_loader : DataLoader
        Test Dataloader.

    path : string
        model path to load the model from

    plot : bool=False
        plot actual vs prediction
    percentage_error_fix : float
        Add a float to the time serie for calculation of percentage because of null values 
    
    Returns
    ----------
    y_pred: array
        predicted values by the model

    y_true : array
        actual values to compare to the prediction
    metric_dict : dictionary
        Contain the following metrics : 
        rmse_val: 
            Root mean square error calculate between y_pred and y_true
        rmspe_val:
            Root mean square percentage error calculate between y_pred and y_true    
        mae_val:
            Mean absolute error calculate between y_pred and y_true
        mape_val:
            Mean absolute percentage error calculate between y_pred and y_true
        maape_val:
            Mean Arctangente percentage error calculate between y_pred and y_true
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.load_state_dict(torch.load(path))
    best_model = best_model.to(device)
    best_model.double()
    best_model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    # test_data = datadict['test_data']
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            x = torch.Tensor(inputs).unsqueeze(1).to(device)
            y = torch.Tensor(targets).unsqueeze(0).to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    test_loss /= len(test_loader)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    y_pred = predictions[:,0]
    y_true = actuals[:,0]
    rmse_val= rmse(y_true,y_pred)
    rmspe_val = rmspe(y_true,y_pred,percentage_error_fix)
    mae_val = mean_absolute_error(y_true ,y_pred)
    mape_val = mape(y_true,y_pred,percentage_error_fix)
    maape_val =  maape(y_true,y_pred,percentage_error_fix)
    metric_dict = {"RMSE":rmse_val, "RMSPE": rmspe_val, "MAE":mae_val,"MAPE":mape_val, "MAAPE": maape_val}

    # Set x and y labels
    if plot : 
        plt.figure(figsize=(28, 5))
        plt.title('Actual vs Prediction')
        plt.plot(y_true, label='Actuals')
        plt.plot(y_pred, label='Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    return y_pred, y_true, metric_dict
