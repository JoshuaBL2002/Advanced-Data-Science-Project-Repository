
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

else:

    st.warning("Please select at least one ASE sensor.")
