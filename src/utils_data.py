
import src.models as models 
import src.utils_graph as gu
from pathlib import Path
import torch 
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    
def ExpSmooth(df_PeMS,alpha=0.2):
    
    """
    Simple Exponential smoothing using the Holt Winters method without using statsmodel
    
    Parameters:
    -----------
    df_PeMS : pd.DataFrame 
        data to smooth
    alpha : float
        exponential smoothing param

    Returns
    -------
    pd.Dataframe
        Dataframe with the input smoothed
    """
    
    # Apply exponential smoothing to the time serie
    for i in range(len(df_PeMS.columns)):
        y = df_PeMS[df_PeMS.columns[i]]
        smoothed_values = [y[0]]
        for j in range(1, len(y)):
            smoothed_value = alpha * y[j] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed_value)
        df_PeMS[df_PeMS.columns[i]] = smoothed_values
    return df_PeMS


def normalize_data(df_PeMS):
    """
    Normalize the data diving by the maximum
    
    Parameters:
    -----------
    df_PeMs : pd.DataFrame 
        data to smooth

    Returns
    -------
    pd.Dataframe
        Dataframe with the input normalized
    """
    maximum = df_PeMS.max().max()
    df_PeMS = df_PeMS /  maximum

    return df_PeMS

#TODO
def createExperimentsData(cluster_size, df_PeMS, layers = 6, perc_train = 0.7, perc_val = 0.15, subgraph = False, overwrite = False):
    import pickle 
    

    """
    Generates pickled (.pkl) dictionary files with the train/val/test data and an associated model

    Parameters
    ----------
    cluster_size : int
        Size of the node clusters

    df_PeMs : pd.Dataframe
        dataframe with all the PeMS data 

    layers: int
        number of layers for the NN model

    perc_train : float
        percentage of the data to be used for training

    perc_test : float
        percentage of the data to be used for testing

    """
    
    train_len = len(df_PeMS)

    if subgraph:
        dirpath = './experiment/cluster'
        subgraph = gu.subgraph_dijkstra(G,i, cluster_size-1)
        nodes_range = range(df_PeMS.columns)
        columns = list(subgraph.nodes)
    else:
        dirpath = './experiments/clusterSubGraph'
        nodes_range = range(len(df_PeMS.columns)+1-cluster_size)
        columns = df_PeMS.columns[i:i+cluster_size]

    filename = Path(dirpath) / f"S{cluster_size}l{train_len}"
    
    if (filename.isfile()):

        
    
        cluster_dict={"size":cluster_size}

        for i in nodes_range:
            model = models.LSTMModel(input_size=cluster_size, hidden_size=32, num_layers=layers, output_size=cluster_size)
            train_loader, val_loader, test_loader = createLoaders(df_PeMS, columns,  perc_train, perc_val)
            cluster_dict[i]={"model":model,"train":train_loader,"val":val_loader,"test":test_loader}

        with open(filename, 'wb') as f:
            pickle.dump(cluster_dict, f)

    return model, train_loader, val_loader, test_loader


    with open('./experiment/clusterS{}.pkl'.format(i), 'rb') as f:
        my_dict = pickle.load(f)
        # iterate on number of cluster 100-i+1
        for j in range(100-i+1):
            train = my_dict[j]["train"]
            val = my_dict[j]["val"]
            model = my_dict[j]["model"]
            model = train_model(model,train, val)
            my_dict[j]["model"]=copy.deepcopy(model)
    with open('./experiment/clusterS{}.pkl'.format(i), 'wb') as f:
        pickle.dump(my_dict, f)
    print('Experiment" + {} +" COMPLETED !'.format(i))

from torch.utils.data import Dataset
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset model with input/target pairs for the LSTM model
    Defines the sliding window size and stride
    """
    
    def __init__(self, data, window_size, stride, target_size=1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.target_size = target_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        inputs = self.data[idx:idx+self.window_size]
        target = self.data[idx+self.window_size:idx+self.window_size+self.target_size]
        return inputs, target
    
def my_data_loader(data, window_size = 7, stride = 1,target_size=1,batch_size=32):
    from torch.utils.data import DataLoader

    """
    Create a Time Serie DataLoader and format it correctly if CUDA is available GPU else CPU

    Parameters
    ----------
    data : pd.Dataframe
        dataframe with all the PeMS data
    windows_size : int
        Sliding window use for training
    stride : int
        the amount of movement after processing each sliding windows
    target_size : int 
        size of the target values of each sliding windows

    """
    
    dataset = TimeSeriesDataset(data.values, window_size, stride, target_size)
    loader = DataLoader(dataset, batch_size, shuffle=False)
    if torch.cuda.is_available():
        loader = [(inputs.to(device), targets.to(device)) for inputs, targets in loader]
    return loader

def createLoaders(df_PeMS, columns=0, perc_train = 0.7, perc_val = 0.15,  window_size = 7, stride = 1, target_size=1, batch_size=32):
    """
    Returns torch.DataLoader for train validation and test data
    
    Parameters
    ----------
    df_PeMs : pd.Dataframe
        dataframe with all the PeMS data
    columns : List 
        List of columns to process
    windows_size : int
        Sliding window use for training
    stride : int
        the amount of movement after processing each sliding windows
    target_size : int 
        size of the target values of each sliding windows
    """
    
    from torch.utils.data import  DataLoader
    
    if columns == 0:
        columns = df_PeMS.columns
        
    train_len = len(df_PeMS)

    train_data= df_PeMS[columns][:int(train_len * perc_train)]
    val_data =  df_PeMS[columns][int(train_len * perc_train): int(train_len * (perc_train + perc_val))]
    test_data = df_PeMS[columns][int(train_len * (perc_train + perc_val)):]
    
    train_loader = my_data_loader(train_data, window_size, stride, target_size, batch_size)
    val_loader = my_data_loader(val_data, window_size, stride, target_size, batch_size)
    test_loader = my_data_loader(test_data, window_size, stride, target_size, batch_size)

    return train_loader, val_loader, test_loader 



def load_PeMS04_flow_data(input_path: Path = "./data/PEMS04/"):
    import pandas as pd
    import numpy as np
    
    """
    
    Function to load traffic flow data from 'npz' and 'csv' files associated with PeMS

    Parameters
    ----------
    input_path: Path
        Path to the input directory

    Returns
    -------
    df_PeMS : pd.Dataframe
        With the flow between two sensors
    
    df_distance:
        Dataframe with the distance metrics between sensors
    
    """


    flow_file = input_path + 'pems04.npz'
    csv_file  = input_path + 'distance.csv'

    # the flow data is stored in 'data' third dimension
    df_flow = np.load(flow_file)['data'][:,:,0]
    df_distance = pd.read_csv(csv_file)
    
    dict_flow = { k : df_flow[:,k] for k in range(df_flow.shape[1])}

    df_PeMS = pd.DataFrame(dict_flow)


    start_date = "2018-01-01 00:00:00"
    end_date = "2018-02-28 23:55:00"
    interval = "5min"
    index = pd.date_range(start=start_date, end=end_date, freq=interval)
    df_PeMS = df_PeMS.set_index(index)

    return df_PeMS, df_distance



def preprocess_PeMS_data(df_PeMS, df_distance, init_node : int = 0, n_neighbors : int = 99):
    from src.utils_graph import create_graph, subgraph_dijkstra
    """
    Filter to n nearest neightbors from 'init_node', sort by mean traffic flow, and normalize and smooth data

    Parameters
    ----------
    init_node : int
        Index of the node we want to start with

    n_neighbors: int
        Number of nearest neighbors to consider
    """

    # Filter nodes to retain only n nearest neighbors
    graph_init = create_graph(df_distance)
    graph_nearest = subgraph_dijkstra(graph_init, init_node, n_neighbors)
    df_PeMS = df_PeMS[list(graph_nearest.nodes)]

    #Sort data hby mean traffic flow

    df_sorted= df_PeMS.mean().sort_values()
    index_mean_flow = df_sorted.index
    column_order = list(index_mean_flow)
    df_PeMS = df_PeMS.reindex(columns = column_order)

    df_PeMS = ExpSmooth(df_PeMS)
    df_PeMS = normalize_data(df_PeMS)

    return df_PeMS