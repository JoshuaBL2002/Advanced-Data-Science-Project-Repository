
# ==============================================
# SECTION 6 - STREAMLIT DASHBOARD
# Advanced Data Science - NO2 Prediction Project
# ==============================================


# ==============================================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================================

# Streamlit is used to create the interactive web dashboard
import streamlit as st

# Pandas is used to read, combine and display CSV datasets
import pandas as pd

# Path is used to locate files and folders in the GitHub repository
from pathlib import Path

# Plotly Express creates interactive graphs
import plotly.express as px


# ==============================================
# 2. CONFIGURE THE STREAMLIT DASHBOARD
# ==============================================

# Configure the browser tab and dashboard layout
# page_title sets the name displayed in the browser tab
# page_icon sets the icon displayed beside the page title
# layout="wide" allows the dashboard to use more screen space

st.set_page_config(
    page_title="NO2 Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ==============================================
# 3. DASHBOARD TITLE AND INTRODUCTION
# ==============================================

# Display the main heading at the top of the dashboard
st.title("Air Quality NO₂ Prediction Dashboard")

# Display a short description underneath the main heading
st.write(
    "Advanced Data Science Project – "
    "ASE Sensor Analysis and Machine Learning"
)

# Add a horizontal divider to separate sections
st.divider()


# ==============================================
# 4. PROJECT OVERVIEW
# ==============================================

# Create a heading for the project introduction
st.header("Project Overview")

# Explain the purpose of the dashboard
# The project investigates NO2 concentrations using
# measurements collected by ASE air quality sensors
# and reference measurements from monitoring stations

st.write(
    "This dashboard explores NO₂ concentrations "
    "recorded by ASE sensors at the SUD3 and QDP "
    "monitoring stations."
)


# ==============================================
# 5. SECTION 6.1 - ASE SENSOR DATA EXPLORER
# ==============================================

# Add a divider before the sensor exploration section
st.divider()

# Display the heading for the data explorer
st.header("ASE Sensor Data Explorer")


# ==============================================
# 5.1 LOCATE THE SENSOR DATASETS
# ==============================================

# Identify the folder containing the nine CSV datasets
# __file__ represents the location of the current app.py file
# .parent selects the folder containing app.py
# / "data" points to the data folder in the GitHub repository

DATA_FOLDER = Path(__file__).parent / "data"


# ==============================================
# 5.2 DEFINE THE AVAILABLE ASE SENSORS
# ==============================================

# Create a dictionary containing the available sensors
# Each sensor name is linked to its corresponding CSV file
# These are the nine datasets used in the project

sensor_files = {
    "ASE4": "ASE4.csv",
    "ASE5": "ASE5.csv",
    "ASE6": "ASE6.csv",
    "ASE7": "ASE7.csv",
    "ASE8": "ASE8.csv",
    "ASE10": "ASE10.csv",
    "ASE11": "ASE11.csv",
    "ASE12": "ASE12.csv",
    "ASE13": "ASE13.csv"
}


# ==============================================
# 5.3 ALLOW USERS TO SELECT MULTIPLE SENSORS
# ==============================================

# Create an interactive multiselect dropdown
# Users can choose one or several ASE sensors
# The available choices come from the sensor_files dictionary
# ASE4 and ASE5 are selected automatically when the app opens

selected_sensors = st.multiselect(
    "Select ASE sensors to compare:",
    options=list(sensor_files.keys()),
    default=["ASE4", "ASE5"]
)


# ==============================================
# 5.4 LOAD THE SELECTED SENSOR DATASETS
# ==============================================

# Check whether the user has selected at least one sensor
# If sensors are selected, their CSV datasets will be loaded
# If no sensors are selected, a warning will be displayed

if selected_sensors:

    # Create an empty list to store the selected datasets
    datasets = []

    # Loop through each sensor selected by the user
    for sensor in selected_sensors:

        # Identify the CSV filename for the current sensor
        # Combine the data folder path with the filename
        file_path = DATA_FOLDER / sensor_files[sensor]

        # Read the sensor CSV file using Pandas
        # sep=";" specifies that columns are separated by semicolons
        # na_values=["NA"] identifies missing measurements
        # Missing values are converted to NaN by Pandas

        df = pd.read_csv(
            file_path,
            sep=";",
            na_values=["NA"]
        )

        # Create a new column called Sensor
        # This identifies which ASE sensor each observation belongs to
        # It is useful when comparing multiple datasets

        df["Sensor"] = sensor

        # Add the current sensor dataset to the datasets list
        datasets.append(df)


    # ==============================================
    # 5.5 COMBINE THE SELECTED DATASETS
    # ==============================================

    # Combine all selected sensor datasets into one DataFrame
    # pd.concat joins the datasets vertically (row by row)
    # ignore_index=True creates a new continuous row index
    # The Sensor column preserves each observation's source

    combined_df = pd.concat(
        datasets,
        ignore_index=True
    )


    # ==============================================
    # 6. SECTION 6.2 - MONITORING STATION SELECTION
    # ==============================================

    # Display the heading for monitoring station selection
    st.subheader("Monitoring Station Selection")

    # Create an interactive dropdown for the monitoring stations
    # Both displays reference measurements from both stations
    # SUD3 selects the SUD3 reference measurements
    # QDP selects the QDP reference measurements

    selected_station = st.selectbox(
        "Select monitoring station:",
        ["Both", "SUD3", "QDP"]
    )


    # ==============================================
    # 6.1 SELECT THE REFERENCE NO2 MEASUREMENTS
    # ==============================================

    # The CSV datasets contain reference NO2 measurements
    # from the SUD3 and QDP monitoring stations
    #
    # Reference_NO2 is a common column name that can
    # represent the selected station's reference measurements
    #
    # This allows later graphs to use the same column name
    # when switching between individual stations

    # If SUD3 is selected, use the SUD3_NO2 column
    if selected_station == "SUD3":

        combined_df["Reference_NO2"] = combined_df["SUD3_NO2"]

    # If QDP is selected, use the QDP_NO2 column
    elif selected_station == "QDP":

        combined_df["Reference_NO2"] = combined_df["QDP_NO2"]

    # If Both is selected, retain both original reference columns
    # Reference_NO2 is set to missing because there is no
    # single reference value selected for each observation
    else:

        combined_df["Reference_NO2"] = pd.NA


    # ==============================================
    # 7. DISPLAY THE SELECTED SENSOR DATA
    # ==============================================

    # Create a heading above the combined dataset
    st.subheader("Selected Sensor Data")

    # Display the combined DataFrame as an interactive table
    # Users can scroll through the rows and columns
    # use_container_width=True fills the available dashboard width

    st.dataframe(
        combined_df,
        use_container_width=True
    )


    # ==============================================
    # 7.1 DISPLAY THE NUMBER OF OBSERVATIONS
    # ==============================================

    # len(combined_df) counts the total number of rows
    # This updates automatically when users select sensors
    # Selecting more sensors generally increases the row count

    st.write(
        "Total observations:",
        len(combined_df)
    )


    # ==============================================
    # 8. DISPLAY REFERENCE NO2 MEASUREMENTS
    # ==============================================

    # Create a separate section for reference measurements
    st.subheader("Reference NO₂ Measurements")

    # If Both is selected, display reference measurements
    # from SUD3 and QDP in separate columns

    if selected_station == "Both":

        # Show the sensor identifier, date and both
        # reference NO2 measurement columns

        st.dataframe(
            combined_df[
                ["Sensor", "date", "SUD3_NO2", "QDP_NO2"]
            ],
            use_container_width=True
        )

    # If an individual station is selected,
    # display the common Reference_NO2 column
    else:

        # Show the sensor identifier, date and
        # reference measurements for the selected station

        st.dataframe(
            combined_df[
                ["Sensor", "date", "Reference_NO2"]
            ],
            use_container_width=True
        )

# ==============================================
# 9. HANDLE NO SENSOR SELECTION
# ==============================================

# This else belongs to the original if selected_sensors statement
# It runs only when the user has not selected any ASE sensors
# Display a warning instead of attempting to load empty datasets


    # ==============================================
    # 10. SECTION 6.3 - INTERACTIVE NO2 GRAPHS
    # ==============================================

    # Display a divider between dashboard sections
    st.divider()

    # Create a heading for the visualisation section
    st.header("Interactive NO₂ Concentration Graphs")

    # Convert the date column into datetime values
    # Invalid dates are converted into missing values
    graph_df = combined_df.copy()

    graph_df["date"] = pd.to_datetime(
        graph_df["date"],
        errors="coerce"
    )

    # Remove rows without a valid date
    graph_df = graph_df.dropna(subset=["date"])

    
    # ==============================================
    # SECTION 6.5 - DATE RANGE FILTER
    # ==============================================

    # Display a heading for date filtering
    st.subheader("Date Range Selection")

    # Identify the earliest and latest dates
    # available in the selected sensor datasets
    min_date = graph_df["date"].min().date()
    max_date = graph_df["date"].max().date()

    # Allow users to select a start and end date
    selected_dates = st.date_input(
        "Select the date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Check whether two valid dates were selected
    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        # Keep measurements within the selected period
        # Comparing dates includes the full final day
        graph_df = graph_df[
            graph_df["date"].dt.date.between(
                start_date,
                end_date
            )
        ].copy()

        # Display the number of remaining observations
        st.write(
            "Observations in selected date range:",
            len(graph_df)
        )

    else:
        # Streamlit may temporarily return one date
        # while the user is choosing a range
        st.info("Please select a start and end date.")

        # Prevent displaying unfiltered graph data
        graph_df = graph_df.iloc[0:0].copy()


    # Convert NO2 measurement columns into numeric values
    # Invalid or missing measurements become NaN
    no2_columns = ["ASE_NO2", "SUD3_NO2", "QDP_NO2"]

    for column in no2_columns:
        graph_df[column] = pd.to_numeric(
            graph_df[column],
            errors="coerce"
        )

    # ==============================================
    # 10.1 SELECT WHICH MEASUREMENTS TO DISPLAY
    # ==============================================

    # Allow users to show or hide the reference stations
    show_reference = st.checkbox(
        "Show reference station measurements",
        value=True
    )

    # ==============================================
    # 10.2 CREATE THE SENSOR COMPARISON GRAPH
    # ==============================================

    # Start with the ASE sensor measurements
    # Each selected sensor will have its own line
    sensor_graph = graph_df[
        ["date", "Sensor", "ASE_NO2"]
    ].copy()

    # Rename the measurement column for plotting
    sensor_graph = sensor_graph.rename(
        columns={"ASE_NO2": "NO2_Value"}
    )

    # Remove rows without a valid NO2 measurement
    sensor_graph = sensor_graph.dropna(
        subset=["NO2_Value"]
    )

    # ==============================================
    # 10.3 ADD REFERENCE STATION MEASUREMENTS
    # ==============================================

    # Create a list containing the sensor graph data
    graph_parts = [sensor_graph]

    if show_reference:

        # Select the reference stations to display
        if selected_station == "Both":
            reference_columns = ["SUD3_NO2", "QDP_NO2"]
        else:
            reference_columns = [selected_station + "_NO2"]

        # Add each reference station as a separate line
        for column in reference_columns:

            # Use one selected sensor's rows for the
            # reference series to avoid duplicating
            # reference values across selected sensors
            reference_df = graph_df[
                graph_df["Sensor"] == selected_sensors[0]
            ][["date", column]].copy()

            # Give the reference series a clear label
            reference_df["Sensor"] = column.replace(
                "_NO2", " Reference"
            )

            # Rename the measurement column
            reference_df = reference_df.rename(
                columns={column: "NO2_Value"}
            )

            # Remove missing reference measurements
            reference_df = reference_df.dropna(
                subset=["NO2_Value"]
            )

            graph_parts.append(reference_df)

    # Combine sensor and reference measurements
    plot_df = pd.concat(
        graph_parts,
        ignore_index=True
    )

    # ==============================================
    # 10.4 DISPLAY THE INTERACTIVE LINE GRAPH
    # ==============================================

    if not plot_df.empty:

        # Create a line graph using Plotly Express
        fig = px.line(
            plot_df.sort_values("date"),
            x="date",
            y="NO2_Value",
            color="Sensor",
            title="NO₂ Concentrations Over Time",
            labels={
                "date": "Date and Time",
                "NO2_Value": "NO₂ Measurement",
                "Sensor": "Sensor / Reference Station"
            }
        )

        # Display the graph in Streamlit
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning(
            "No valid NO₂ measurements are available "
            "for the selected sensors."
        )

    # ==============================================
    # 11. SECTION 6.4 - MODEL COMPARISON
    # ==============================================

    # Add a divider between dashboard sections
    st.divider()

    # Display the model comparison heading
    st.header("Machine Learning Model Comparison")

    # Explain the purpose of the comparison
    st.write(
        "Compare the performance of four regression "
        "models used to predict reference NO₂ measurements."
    )

    # ==============================================
    # 11.1 LOAD MODEL EVALUATION RESULTS
    # ==============================================

    # Identify the CSV containing the model results
    # This file is exported from Section 5 in Colab
    results_path = DATA_FOLDER / "model_results.csv"

    # Check whether the results file exists
    if results_path.exists():

        # Load the evaluation results using Pandas
        results_df = pd.read_csv(results_path)

        # ==============================================
        # 11.2 DISPLAY MODEL RESULTS TABLE
        # ==============================================

        # Display the evaluation metrics for each model
        st.subheader("Model Evaluation Results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # ==============================================
        # 11.3 SELECT MODELS TO COMPARE
        # ==============================================

        # Allow users to select one or more models
        # The options come from the results CSV
        selected_models = st.multiselect(
            "Select regression models to compare:",
            options=results_df["Model"].tolist(),
            default=results_df["Model"].tolist()
        )

        # Filter the table to selected models
        filtered_results = results_df[
            results_df["Model"].isin(selected_models)
        ]

        # ==============================================
        # 11.4 SELECT EVALUATION METRIC
        # ==============================================

        # Allow users to choose which metric to plot
        selected_metric = st.selectbox(
            "Select evaluation metric:",
            ["MAE", "RMSE", "R2"]
        )

        # ==============================================
        # 11.5 CREATE MODEL COMPARISON GRAPH
        # ==============================================

        # Only create a graph if models are selected
        if not filtered_results.empty:

            # Create an interactive bar chart
            fig_models = px.bar(
                filtered_results,
                x="Model",
                y=selected_metric,
                color="Model",
                title="Regression Model Performance",
                labels={
                    "Model": "Regression Model",
                    selected_metric: selected_metric
                }
            )

            # Display the chart in Streamlit
            st.plotly_chart(
                fig_models,
                use_container_width=True
            )

        else:
            st.warning(
                "Please select at least one model."
            )

        # ==============================================
        # 11.6 IDENTIFY THE BEST PERFORMING MODEL
        # ==============================================

        # RMSE is used to identify the lowest error
        # A lower RMSE indicates better prediction accuracy
        # for the evaluation data used in Section 5

        valid_results = results_df.dropna(
            subset=["RMSE"]
        )

        if not valid_results.empty:

            best_model = valid_results.loc[
                valid_results["RMSE"].idxmin()
            ]

            # Display the model with the lowest RMSE
            st.success(
                "Lowest RMSE: "
                + str(best_model["Model"])
            )

            # Display its RMSE value
            st.write(
                "RMSE:",
                round(best_model["RMSE"], 4)
            )

        else:
            st.info(
                "Add the actual model evaluation "
                "results from Section 5."
            )

    else:
        st.warning(
            "model_results.csv has not been uploaded "
            "to the data folder yet."
        )


else:

    st.warning("Please select at least one ASE sensor.")
