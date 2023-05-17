# Import necessary library
import folium 
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium, folium_static
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import osmnx as ox
import networkx as nx

###############################################################################
# Utility functions
###############################################################################
# TODO move the function in a utility file
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# TODO move the function in a utility file 
def create_circle_precision_predict(marker_location, value_percent, map_folium, color):
    """
    Draw a circle at the position of the marker.

    Parameters:
    ----------
        marker_location (Marker Folium)

        value_percent (float)
            
        map_folium (Map Folium)

        color : 
            Hex code HTML
    """
    lat, long = marker_location
    # folium.Circle(location=[lat+0.0020,long+0.0018], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    # folium.Circle(location=[lat+0.0020,long+0.0018], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)
    # folium.map.Marker([lat+0.0020,long+0.0016], icon=folium.features.DivIcon(html=f"<div style='font-size: 10pt; color: black'>{int(value_percent*100)}%</div>")).add_to(map_folium)
    folium.Circle(location=[lat,long], color="black", radius=100, fill=True, opacity=1, fill_opacity=0.8, fill_color="white").add_to(map_folium)
    folium.Circle(location=[lat,long], color=color, radius=100*value_percent, fill=True, opacity=0, fill_opacity=1, fill_color=color).add_to(map_folium)


###############################################################################
# Data
###############################################################################
# TODO Try to find a way to get [lat, long] on PeMS04
# Define the latitude and longitude coordinates of Seattle roads 
seattle_roads = {
    "Captor_01": [47.679470, -122.315626],
    "Captor_02": [47.679441, -122.306665],
    "Captor_03": [47.683058163418266, -122.30074031156877],
    "Captor_04": [47.67941986097163, -122.29031294544225],
    "Captor_05": [47.67578888921566, -122.30656814568495],
    "Captor_06": [47.67575649888934, -122.29026613694701],
    "Captor_07": [47.68307457244817, -122.29054200791231],
    "Captor_08": [47.68300028244276, -122.3121427044953],
    "Captor_09": [47.670728396123444, -122.31192781883172],
    "Captor_10": [47.675825, -122.315658],
    "Captor_11": [47.69132417321706, -122.31221442807933],
    "Captor_12": [47.68645681961068, -122.30076590191602],
    "Captor_13": [47.68304467808857, -122.27975989945097],
    "Captor_14": [47.6974488132659, -122.29057907732675]
}

# TODO use adjacency matrix
# Shape (len(road)*len(road)) with 0 if no link between the road or 1 if link between the road
# Add polylines to connect the roads
polyline_roads = [("Captor_01", "Captor_10"), ("Captor_02", "Captor_01"), ("Captor_02", "Captor_04"), ("Captor_02", "Captor_05"),
                ("Captor_03", "Captor_12"), ("Captor_04", "Captor_07"), ("Captor_05", "Captor_06"), ("Captor_05", "Captor_10"), ("Captor_07", "Captor_08"), ("Captor_07", "Captor_13"), ("Captor_07", "Captor_14"),
                ("Captor_08", "Captor_09"), ("Captor_08", "Captor_11")]


###############################################################################
# Map Folium
###############################################################################
# TODO find the global location of captors of PeMS04
# Create maps centered on Seattle
seattle_map = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)
seattle_map_global = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)
seattle_map_local = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)

st.set_page_config(layout="wide")
st.title('Analyses results experimentation')

for road, coords in seattle_roads.items():
    tooltip = f"Road: {road}"
    folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map)
    folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map_global)
    folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="lightgray")).add_to(seattle_map_local)

for start, end in polyline_roads:
    locations = [seattle_roads[start], seattle_roads[end]]
    folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map)
    folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map_global)
    folium.PolyLine(locations=locations, weight=3, color="brown").add_to(seattle_map_local)

for road, coords in seattle_roads.items():
    tooltip = f"Road: {road}"
    color = "green" if road in ["Captor_07", "Captor_08"] else "red" if road == "Captor_03" else "blue"
    if road == "Captor_07" :
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
        create_circle_precision_predict(coords, 0.90, seattle_map_global, "#22ED1E")
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
        create_circle_precision_predict(coords, 0.87*0.90, seattle_map_local, "#C92A2A")
    elif road == "Captor_08" :
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
        create_circle_precision_predict(coords, 0.85, seattle_map_global, "#4B4BD9")
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
        create_circle_precision_predict(coords, 0.85, seattle_map_local, "#4B4BD9")
    elif road == "Captor_03" :
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_global)
        create_circle_precision_predict(coords, 0.90, seattle_map_global, "#22ED1E")
        folium.Marker(location=coords, tooltip=tooltip, icon=folium.Icon(color="black")).add_to(seattle_map_local)
        create_circle_precision_predict(coords, 0.89*0.90, seattle_map_local, "#C92A2A")

# Center the map
seattle_map.fit_bounds(seattle_map.get_bounds())
seattle_map_global.fit_bounds(seattle_map_global.get_bounds())
seattle_map_local.fit_bounds(seattle_map_local.get_bounds())


###############################################################################
# Streamlint
###############################################################################
# Container for the general map
with st.container():
    col1, col2, col3 = st.columns((1,2,1))
    with col2:
        col2.header("Seattle Map")
        folium_static(seattle_map, width=750)

# Create a table
col1, col2 = st.columns(2, gap="small")
with col1:
    col1.header('Federated model results')
    folium_static(seattle_map_global, width=650)
with col2:
    col2.header('Local models results')
    folium_static(seattle_map_local, width=650)









###############################################################################
# Statistics
###############################################################################
###############################################################################
# Find the data
###############################################################################
dossier = os.listdir('./center_and_reduce') # liste le contenu du dossier courant

dossiers = list(dossier)
for i in dossiers:
    nb_captors = [name.split("_")[2] for name in dossiers]

# Créer un sélecteur pour choisir le modèle à afficher
nb_captor = st.selectbox('Choose the number of captor', nb_captors)

dossier = os.listdir(f"./center_and_reduce/nb_captor_{nb_captor}")
dossiers = list(dossier)

for _ in dossiers:
    windows_horizons = [(name.split("_")[1], name.split("_")[3]) for name in dossier]

window_horizon = st.selectbox('Choose the couple windows and horizon size', windows_horizons)

normalization = "center_and_reduce"
path_save_model = f"./{normalization}/nb_captor_{nb_captor}/windows_{window_horizon[0]}_out_{window_horizon[1]}"

final_results = pd.read_pickle(f"{path_save_model}/final_resultats.pkl")

###############################################################################
# Choose the statistics
###############################################################################
results = st.selectbox('Choose what you want to see', ["all", "general", "diff", "good and bad", "boxplot"])

if(results in ["good and bad", "boxplot"]):
    st.header("Focus on MAAPE values")
    st.text("For MAAPE, the smaller the value is, the better the model performs")
    st.text("MAAPE values are in percent (%)")
    st.text("A positive value means that the model compare to the other perform x % better")
    st.text("A negative value means that the model compare to the other perform x % worse")


###############################################################################
# Print statistics and plots
###############################################################################
model_names = ["TGCN Model", "LSTM Model", "GRU Model"]


    ###############################################################################
    # General
    ###############################################################################
if(results == "all" or results == "general"):
    st.subheader('Summary statistics of how well the predictions goes (values are in %)')
    st.text("For RMSE and MAAPE, the smaller the value is, the better the model performs")
    st.dataframe(final_results.groupby("Model_Name")[["RMSE", "MAAPE"]].describe().rename(columns={"count": "nb captor"}), use_container_width=True)


    ###############################################################################
    # Diff
    ###############################################################################
diff_resultats = []
# Compute results
for i in final_results.index.get_level_values("Captor").unique():
    diff_resultats.extend(
        (
            [
                "TGCN vs LSTM",
                round(
                    final_results.loc[i].loc["TGCN Model"]["MAAPE"]
                    - final_results.loc[i].loc["LSTM Model"]["MAAPE"],
                    2,
                ),
            ],
            [
                "TGCN vs GRU",
                round(
                    final_results.loc[i].loc["TGCN Model"]["MAAPE"]
                    - final_results.loc[i].loc["GRU Model"]["MAAPE"],
                    2,
                ),
            ],
            [
                "LSTM vs TGCN",
                round(
                    final_results.loc[i].loc["LSTM Model"]["MAAPE"]
                    - final_results.loc[i].loc["TGCN Model"]["MAAPE"],
                    2,
                ),
            ],
            [
                "LSTM vs GRU",
                round(
                    final_results.loc[i].loc["LSTM Model"]["MAAPE"]
                    - final_results.loc[i].loc["GRU Model"]["MAAPE"],
                    2,
                ),
            ],
            [
                "GRU vs TGCN",
                round(
                    final_results.loc[i].loc["GRU Model"]["MAAPE"]
                    - final_results.loc[i].loc["TGCN Model"]["MAAPE"],
                    2,
                ),
            ],
            [
                "GRU vs LSTM",
                round(
                    final_results.loc[i].loc["GRU Model"]["MAAPE"]
                    - final_results.loc[i].loc["LSTM Model"]["MAAPE"],
                    2,
                ),
            ],
        )
    )

diff_resultats = pd.DataFrame(diff_resultats, columns=["Diff", "MAAPE"])
diff_resultats.set_index(["Diff"], inplace=True)

if(results == "all" or results == "diff"):
    st.header("Focus on MAAPE values")
    st.text("For MAAPE, the smaller the value is, the better the model performs")
    st.subheader('Summary statistics of how a model perform better than an other (values are in %)')
    st.text("A positive value means that the model compare to the other perform x % better")
    st.text("A negative value means that the model compare to the other perform x % worse")
    diff_resultats = diff_resultats * -1
    st.dataframe(diff_resultats.groupby("Diff").describe().rename(columns={"count": "nb captor"}), use_container_width=True)


    ###############################################################################
    # Good and bad
    ###############################################################################
# Initialisation dict good and bad results
good_results = {model_name: [] for model_name in model_names}
bad_results = {model_name: [] for model_name in model_names}


stats = {model_name: {"nb_win": 0} for model_name in model_names}
for captor_id in final_results.index.get_level_values("Captor").unique():
    best_model_name = final_results.loc[captor_id]["MAAPE"].idxmin()
    best_model_result = final_results.loc[captor_id].loc[best_model_name]["MAAPE"]
    stats[best_model_name]["nb_win"] += 1
    good_results[best_model_name].append(best_model_result)
    for model_name in model_names:
        if model_name != best_model_name:
            bad_result = final_results.loc[captor_id].loc[model_name]["MAAPE"]
            bad_results[model_name].append(bad_result)
            
good_df = {}
bad_df = {}
for model_name in model_names:
    good_df[model_name] = pd.DataFrame(good_results[model_name], columns=[f"{model_name}"])
    bad_df[model_name] = pd.DataFrame(bad_results[model_name], columns=[f"{model_name}"])


if(results == "all" or results == "good and bad"):
    col1, col2 = st.columns(2) 
    col1.header('Summary where the model get the better results')
    col2.header('Summary where the model not get the best results')
    for model_name in model_names:
        col1.dataframe(good_df[model_name].describe(), use_container_width=True) 
        col2.dataframe(bad_df[model_name].describe(), use_container_width=True)


    ###############################################################################
    # Boxplot
    ###############################################################################
if(results == "all" or results == "boxplot"):
    # print boxplot
    st.subheader('Box plot of MAAPE values by models')
    st.text("For MAAPE, the smaller the value is, the better the model performs")
    fig, ax = plt.subplots()
    bar_plot_results = final_results.reset_index()
    bar_plot_results.boxplot(column="MAAPE", by="Model_Name", fontsize=10, figsize=(8,8), ylabel="MAAPE values", xlabel="Model Name", ax=ax)
    plt.yticks(np.arange(0, bar_plot_results["MAAPE"].max(), 4))
    st.pyplot(fig)