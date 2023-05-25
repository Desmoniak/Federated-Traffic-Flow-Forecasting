###############################################################################
# Libraries
###############################################################################
import os
import glob

import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#######################################################################
# Utility
#######################################################################
def result_prediction(predictions, actuals):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    
    indices_by_month = []
    EPSILON = 1e-5
    # Créer une liste vide pour stocker les données du tableau
    data = []
    y_pred = predictions[:]
    y_true = actuals[:]

    signe = "-" if np.mean(y_pred - y_true) < 0 else "+"
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))*100
    maape =  np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + EPSILON))))*100
    
    return [signe, mae, rmse, smape, maape]

def add_to_results(df_results, n_neighbors, result_dict, model_name, df_PeMS):
    for i in range(n_neighbors):
        temp = result_prediction(result_dict["pred_value"][:,:,i],result_dict["true_value"][:,:,i])
        temp.append(model_name)
        temp.append(f"captor {df_PeMS.columns[i]}")
        df_results.append(temp)
    return df_results

def compute_percentage_change(df, metric):
    df_final = pd.DataFrame()
    df_index = df.index
    for index_one in df_index:
        for index_two in df_index:
            if(index_one != index_two):
                temp = \
                pd.DataFrame(
                    (((final_results_groupby_captor.loc[index_one].loc[metric] - final_results_groupby_captor.loc[index_two].loc[metric]) / (final_results_groupby_captor.loc[index_two].loc[metrics_ratio]).abs()) * 100)).T
                temp.index = [f"(({index_one} - {index_two}) / {index_two}) * 100"]
                temp.index.name = f"{metric}"
                temp = temp[["mean","std"]]
                df_final = pd.concat([df_final, temp],axis=0) 
    return df_final



#######################################################################
# Loading Data
#######################################################################
normalization = "center_and_reduce"
if files := glob.glob(f"./{normalization}/**/**/*dict.pkl"):
    nb_captors = [i.split("\\")[1].split("_")[2] for i in files]
    nb_captor = st.selectbox('Choose the number of captor', set(nb_captors))

    windows_horizons = [(name.split("_")[1], name.split("_")[3]) for name in list(os.listdir(f"./center_and_reduce/nb_captor_{nb_captor}"))]
    window_horizon = st.selectbox('Choose the couple windows and horizon size', windows_horizons)

    path_save_model = f"./{normalization}/nb_captor_{nb_captor}/windows_{window_horizon[0]}_out_{window_horizon[1]}"
    tgcn_dict = pd.read_pickle(f"{path_save_model}/tgcn_dict.pkl")
    lstm_dict = pd.read_pickle(f"{path_save_model}/lstm_dict.pkl")
    gru_dict = pd.read_pickle(f"{path_save_model}/gru_dict.pkl")


#######################################################################
# Preprocessing + DATA
#######################################################################
    from src.utils_data import load_PeMS04_flow_data, preprocess_PeMS_data
    final_resultats = []
    n_neighbors = int(nb_captor)
    df_PeMS_old, df_distance  = load_PeMS04_flow_data()
    df_PeMS, adjacency_matrix_PeMS, meanstd_dict = preprocess_PeMS_data(df_PeMS_old, df_distance, init_node=0, n_neighbors=n_neighbors, center_and_reduce=True)

    final_resultats = add_to_results(final_resultats, n_neighbors, tgcn_dict, "TGCN Model", df_PeMS)
    final_resultats = add_to_results(final_resultats, n_neighbors, lstm_dict, "LSTM Model", df_PeMS)
    final_resultats = add_to_results(final_resultats, n_neighbors, gru_dict, "GRU Model", df_PeMS)
    final_resultats = pd.DataFrame(final_resultats, columns=['Signe error', 'MAE', 'RMSE', 'SMAPE', 'MAAPE', "Model_Name", "Captor"])

    final_results_captor_models = final_resultats.set_index(["Captor","Model_Name"]).sort_values(by=['Captor', "Model_Name"])
    model_names = final_results_captor_models.index.get_level_values("Model_Name").unique()

    final_results_groupby_captor = final_resultats.groupby("Model_Name").describe()
    metrics = list(final_results_groupby_captor.columns.get_level_values(0).unique())


#######################################################################
# STREAMLINT
#######################################################################

    #######################################################################
    # Statistics for models on each captor
    #######################################################################
    c1_captor_models, c2_group_by_models = st.columns(2, gap="large")
    with c1_captor_models:
        st.header("Metrics for each captors on every models")
        st.dataframe(final_results_captor_models, use_container_width=True)
    #######################################################################
    # Statistics global for each model
    #######################################################################
    with c2_group_by_models:
        st.header("Summary metrics for each models on all captors")
        metric = st.selectbox('Select the metric', metrics)
        for model in model_names:
                st.dataframe(pd.DataFrame((final_results_groupby_captor.loc[model].loc[metric])).T, use_container_width=True)


        #######################################################################
        # Evolution in percent
        #######################################################################
    c1_percentage_change, c2_percentage_change, c3_percentage_change = st.columns((1,2,1))
    with c2_percentage_change:
        c2_percentage_change.header("Evolution (%)")
        metrics_ratio = st.selectbox('Select the metric', metrics, key="metrics_ratio") # Créer un sélecteur pour choisir le modèle à afficher
        st.dataframe(compute_percentage_change(final_results_groupby_captor, metrics_ratio), use_container_width=True)


        #######################################################################
        # Plots
        #######################################################################
    col1_boxplot, col2_boxplot, col3_boxplot = st.columns((1,2,1))
    with col2_boxplot:
        metric_boxplot = st.selectbox('Select the metric', metrics, key="metric_boxplot") # Créer un sélecteur pour choisir le modèle à afficher
        st.subheader(f'Box plot of {metric_boxplot} values by models')
        fig, ax = plt.subplots()
        bar_plot_results = final_resultats
        bar_plot_results.boxplot(column=f"{metric_boxplot}", by="Model_Name", ylabel=f"{metric_boxplot} values", xlabel="Model Name", ax=ax)
        plt.yticks(np.arange(0, bar_plot_results[f"{metric_boxplot}"].max(), 10))
        st.pyplot(fig, use_container_width=True)


        ###############################################################################
        # Good and bad
        ###############################################################################
    metric_good_bad = st.selectbox('Select the metric', metrics, key="metrics_good_bad")
    
    good_results = {model_name: [] for model_name in model_names}
    bad_results = {model_name: [] for model_name in model_names}

    for captor_id in final_results_captor_models.index.get_level_values("Captor").unique():
        best_model_name = final_results_captor_models.loc[captor_id]["RMSE"].idxmin()
        good_results[best_model_name].append(final_results_captor_models.loc[captor_id].loc[best_model_name]["RMSE"])
        for model_name in model_names:
            if model_name != best_model_name:
                bad_result = final_results_captor_models.loc[captor_id].loc[model_name]["RMSE"]
                bad_results[model_name].append(bad_result)
                    
    col_good, col_bad = st.columns(2, gap="medium")
    col_good.header('Summary where the model get the better result')
    col_bad.header('Summary where the model don\'t get the best result')
    for model_name in model_names:
        with col_good:
            col_good.subheader(f"{model_name}")
            good_results[model_name] = pd.DataFrame(good_results[model_name], columns=[f"{model_name}"])
            good_results[model_name] = good_results[model_name].describe()
            good_results[model_name].index.name = metric_good_bad
            st.dataframe(good_results[model_name], use_container_width=True)
        with col_bad:
            col_bad.subheader(f"{model_name}")
            bad_results[model_name] = pd.DataFrame(bad_results[model_name], columns=[f"{model_name}"])
            bad_results[model_name] = bad_results[model_name].describe()
            bad_results[model_name].index.name = metric_good_bad
            st.dataframe(bad_results[model_name], use_container_width=True)





























# ###############################################################################
# # Utility functions
# ###############################################################################
# def create_circle_precision_predict(marker_location, value_percent, map_folium, color):
#     """
#     Draw a circle at the position of the marker.

#     Parameters:
#     ----------
#         marker_location (Marker Folium)

#         value_percent (float)
            
#         map_folium (Map Folium)

#         color : 
#             Hex code HTML
#     """
#     lat, long = marker_location
#     folium.Circle(location=[lat+0.0020,long+0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
#     folium.Circle(location=[lat+0.0020,long+0.0018], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)
#     folium.map.Marker([lat+0.0022,long+0.0014], icon=folium.features.DivIcon(html=f"<div style='font-weight:bold; font-size: 15pt; color: black'>{int(value_percent*100)}%</div>")).add_to(map_folium)
#     # folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
#     # folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


# ###############################################################################
# # Data
# ###############################################################################
# # TODO Try to find a way to get [lat, long] on PeMS04
# # Define the latitude and longitude coordinates of Seattle roads 
# seattle_roads = {
#     "Captor_01": [47.679470, -122.315626],
#     "Captor_02": [47.679441, -122.306665],
#     "Captor_03": [47.683058163418266, -122.30074031156877],
#     "Captor_04": [47.67941986097163, -122.29031294544225],
#     "Captor_05": [47.67578888921566, -122.30656814568495],
#     "Captor_06": [47.67575649888934, -122.29026613694701],
#     "Captor_07": [47.68307457244817, -122.29054200791231],
#     "Captor_08": [47.68300028244276, -122.3121427044953],
#     "Captor_09": [47.670728396123444, -122.31192781883172],
#     "Captor_10": [47.675825, -122.315658],
#     "Captor_11": [47.69132417321706, -122.31221442807933],
#     "Captor_12": [47.68645681961068, -122.30076590191602],
#     "Captor_13": [47.68304467808857, -122.27975989945097],
#     "Captor_14": [47.6974488132659, -122.29057907732675]
# }

# # TODO use adjacency matrix
# # Shape (len(road)*len(road)) with 0 if no link between the road or 1 if link between the road
# # Add polylines to connect the roads
# polyline_roads = [("Captor_01", "Captor_10"), ("Captor_02", "Captor_01"), ("Captor_02", "Captor_04"), ("Captor_02", "Captor_05"),
#                 ("Captor_03", "Captor_12"), ("Captor_04", "Captor_07"), ("Captor_05", "Captor_06"), ("Captor_05", "Captor_10"), ("Captor_07", "Captor_08"), ("Captor_07", "Captor_13"), ("Captor_07", "Captor_14"),
#                 ("Captor_08", "Captor_09"), ("Captor_08", "Captor_11")]


# ###############################################################################
# # Map Folium
# ###############################################################################
# # TODO find the global location of captors of PeMS04
# # Create maps centered on Seattle
# seattle_map = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)
# seattle_map_global = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)
# seattle_map_local = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)

# st.set_page_config(layout="wide")
# st.title('Analyses results experimentation')

# for road, coords in seattle_roads.items():
#     tooltip = f"Road: {road}"
#     folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map)
#     folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map_global)
#     folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map_local)

# for start, end in polyline_roads:
#     locations = [seattle_roads[start], seattle_roads[end]]
#     folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map)
#     folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map_global)
#     folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map_local)

# for road, coords in seattle_roads.items():
#     tooltip = f"Road: {road}"
#     color = "green" if road in ["Captor_07", "Captor_08"] else "red" if road == "Captor_03" else "blue"
#     if road == "Captor_07" :
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
#         create_circle_precision_predict(coords, 0.90, seattle_map_global, "#22ED1E")
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
#         create_circle_precision_predict(coords, 0.87*0.90, seattle_map_local, "#E54640")
#     elif road == "Captor_08" :
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
#         create_circle_precision_predict(coords, 0.85, seattle_map_global, "#6EBEFF")
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
#         create_circle_precision_predict(coords, 0.85, seattle_map_local, "#6EBEFF")
#     elif road == "Captor_03" :
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
#         create_circle_precision_predict(coords, 0.90, seattle_map_global, "#22ED1E")
#         folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
#         create_circle_precision_predict(coords, 0.89*0.90, seattle_map_local, "#E54640")

# # Center the map according to the markers on the map
# seattle_map.fit_bounds(seattle_map.get_bounds())
# seattle_map_global.fit_bounds(seattle_map_global.get_bounds())
# seattle_map_local.fit_bounds(seattle_map_local.get_bounds())


# ###############################################################################
# # Streamlint
# ###############################################################################
# # Container for the general map
# with st.container():
#     col1, col2, col3 = st.columns((1,2,1))
#     with col2:
#         col2.header("Seattle Map")
#         folium_static(seattle_map, width=750)

# # Create a table
# col1, col2 = st.columns(2, gap="small")
# with col1:
#     col1.header('Federated model results')
#     folium_static(seattle_map_global, width=650)
# with col2:
#     col2.header('Local models results')
#     folium_static(seattle_map_local, width=650)







