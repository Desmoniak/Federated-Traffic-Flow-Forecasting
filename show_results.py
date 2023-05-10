# Importer les librairies nécessaires
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Créer un titre pour le dashboard
st.title('Analyses results experimentation')

import os
dossier = os.listdir('./center_and_reduce') # liste le contenu du dossier courant

dossiers = list(dossier)
for i in dossiers:
    nb_captors = [name.split("_")[2] for name in dossiers]

# Créer un sélecteur pour choisir le modèle à afficher
nb_captor = st.selectbox('Choose the number of captor', nb_captors)

dossier = os.listdir(f"./center_and_reduce/nb_captor_{nb_captor}")
dossiers = list(dossier)
for i in dossiers:
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
# _window_size = 7
# horizon = 1
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

diff_resultats = pd.DataFrame(diff_resultats, columns=["Diff", "RMSE"])
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
    st.subheader('Box plot of RMSE values by models')
    st.text("For RMSE, the smaller the value is, the better the model performs")
    fig, ax = plt.subplots()
    bar_plot_results = final_results.reset_index()
    bar_plot_results.boxplot(column="MAAPE", by="Model_Name", fontsize=10, figsize=(8,8), ylabel="MAAPE values", xlabel="Model Name", ax=ax)
    plt.yticks(np.arange(0, bar_plot_results["MAAPE"].max(), 4))
    st.pyplot(fig)
###############################################################################