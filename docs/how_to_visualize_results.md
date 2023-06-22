# How to visualize your results

* First create your config.json for your experiment. /!\ IMPORTANT for the "save_model_path" parameter it's the folder name so don't forget the "/" at the end of the name.
* Second just start the experiment like this:
    * ```bash
        python.exe ./main.py ./[your config.json]
        ```
    * The command will create the folder of your experiment in ./experiments/[here the path to your expriment that you choose in the config.json]. In this folder a file with exactly this name "config.json" will be created and some others files.
* Third test the experiment like this:
    * ```bash
        python.exe test.py ./experiments/[here the path to your experiment]/config.json
        ```
* Fourth start streamlit to visualize like this:
    * ``` bash
        streamlit.exe run ./src/Home.py
        ```