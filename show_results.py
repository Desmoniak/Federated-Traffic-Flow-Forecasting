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

st.set_page_config(layout="wide")
# Créer un titre pour le dashboard
st.title('Analyses results experimentation')

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

# Add polylines to connect the roads
polyline_roads = [("Captor_01", "Captor_10"), ("Captor_02", "Captor_01"), ("Captor_02", "Captor_04"), ("Captor_02", "Captor_05"),
                ("Captor_03", "Captor_12"), ("Captor_04", "Captor_07"), ("Captor_05", "Captor_06"), ("Captor_05", "Captor_10"), ("Captor_07", "Captor_08"), ("Captor_07", "Captor_13"), ("Captor_07", "Captor_14"),
                ("Captor_08", "Captor_09"), ("Captor_08", "Captor_11")]



# Create a map centered on Seattle
seattle_map_global = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)
seattle_map_local = folium.Map(location=[47.67763404920509, -122.30064862690335], zoom_start=15)

def create_bars_true_pred_value(marker_location, true_value, pred_value, map_folium):
    """
    Create two bars in the top-right corner near the marker location.
    
    Parameters
    ----------
        marker_location : 
            coordinate of the Marker Filium [lat, long]
        true_value : 
            the true value
        pred_value : 
            the value predict by the model 
        map_folium : 
            the map folium where bars will be added
    """
    lat, long = marker_location
    max_value = max(true_value, pred_value)
    folium.Rectangle(bounds=[[lat + 0.001, long + 0.0015], [lat + 0.0039*(pred_value/max_value), long + 0.0024]], tooltip=f"Pred value : {pred_value}", color="black", radius=30, fill=True, opacity=100, fill_opacity=1, fill_color="orange").add_to(map_folium)
    folium.Rectangle(bounds=[[lat + 0.001, long + 0.0030], [lat + 0.0039*(true_value/max_value), long + 0.0039]], tooltip=f"True value : {true_value}", color="black", radius=30, fill=True,opacity=100, fill_opacity=1, fill_color="brown").add_to(map_folium)

###############################################################################
# Map Global
###############################################################################
for road, coords in seattle_roads.items():
    tooltip = f"Road: {road}"
    color = "green" if road in ["Captor_07", "Captor_08"] else "red" if road == "Captor_03" else "blue"
    if road in ["Captor_07", "Captor_08"] : 
        radius = 130 
    elif road == "Captor_03" :
        radius = 80
    folium.Marker(location=coords, tooltip=tooltip).add_to(seattle_map_global)
    if road in ["Captor_03", "Captor_07", "Captor_08"] : 
        folium.Circle(location=coords, color=color, radius=radius, fill=True, opacity=100, fill_opacity=1, fill_color=color).add_to(seattle_map_global)
        create_bars_true_pred_value(coords, 230, 500, seattle_map_global)

for start, end in polyline_roads:
    locations = [seattle_roads[start], seattle_roads[end]]
    folium.PolyLine(locations=locations, weight=3, color="blue").add_to(seattle_map_global)

seattle_map_global.fit_bounds(seattle_map_global.get_bounds())


###############################################################################
# Map Local
###############################################################################
for road, coords in seattle_roads.items():
    tooltip = f"Road: {road}"
    color = "red" if road in ["Captor_07", "Captor_08"] else "green" if road == "Captor_03" else "blue"
    if road in ["Captor_07", "Captor_08"] : 
        radius = 70  
    elif road == "Captor_03" :
        radius = 50
    folium.Marker(location=coords, tooltip=tooltip).add_to(seattle_map_local)
    if road in ["Captor_03", "Captor_07", "Captor_08"] : 
        folium.Circle(location=coords, color=color, radius=radius, fill=True, opacity=100, fill_opacity=1, fill_color=color).add_to(seattle_map_local)
        create_bars_true_pred_value(coords, 100, 78, seattle_map_local)

for start, end in polyline_roads:
    locations = [seattle_roads[start], seattle_roads[end]]
    folium.PolyLine(locations=locations, weight=3, color="blue").add_to(seattle_map_local)

seattle_map_local.fit_bounds(seattle_map_local.get_bounds())

col1, col2 = st.columns(2, gap="medium")
with col1:
    col1.header('global model results')
    folium_static(seattle_map_local, width=750)
with col2:
    col2.header('Local model results')
    folium_static(seattle_map_global, width=750)

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

results = st.selectbox('Choose what you want to see', ["all", "general", "diff", "good and bad", "boxplot"])

if(results in ["good and bad", "boxplot"]):
    st.header("Focus on MAAPE values")
    st.text("For MAAPE, the smaller the value is, the better the model performs")
    st.text("MAAPE values are in percent (%)")
    st.text("A positive value means that the model compare to the other perform x % better")
    st.text("A negative value means that the model compare to the other perform x % worse")

# Parameters
normalization = "center_and_reduce"
path_save_model = f"./{normalization}/nb_captor_{nb_captor}/windows_{window_horizon[0]}_out_{window_horizon[1]}"

model_names = ["TGCN Model", "LSTM Model", "GRU Model"]
diff_resultats = []
###############################################################################
# Filtrer les résultats selon le modèle choisi
final_results = pd.read_pickle(f"{path_save_model}/final_resultats.pkl")

if(results == "all" or results == "general"):
    st.subheader('Summary statistics of how well the predictions goes (values are in %)')
    st.text("For RMSE and MAAPE, the smaller the value is, the better the model performs")
    st.dataframe(final_results.groupby("Model_Name")[["RMSE", "MAAPE"]].describe().rename(columns={"count": "nb captor"}), use_container_width=True)
###############################################################################

###############################################################################
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
###############################################################################