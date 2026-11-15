
import streamlit as st
import pandas as pd
from pathlib import Path

# Configure the dashboard
st.set_page_config(
    page_title="NO2 Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Dashboard heading
st.title("Air Quality NO₂ Prediction Dashboard")

st.write(
    "Advanced Data Science Project – "
    "ASE Sensor Analysis and Machine Learning"
)

st.divider()

st.header("Project Overview")

st.write(
    "This dashboard explores NO₂ concentrations "
    "recorded by ASE sensors at the SUD3 and QDP "
    "monitoring stations."
)

st.info("Sensor comparisons and model results will be added next.")


# =====================================
# Section 6.1 - ASE Sensor Data Explorer
# =====================================

st.divider()
st.header("ASE Sensor Data Explorer")

# Location of the CSV datasets
DATA_FOLDER = Path(__file__).parent / "data"

# Available sensor files
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

# Allow multiple sensors to be selected
selected_sensors = st.multiselect(
    "Select ASE sensors to compare:",
    options=list(sensor_files.keys()),
    default=["ASE4", "ASE5"]
)

# Load the selected datasets
if selected_sensors:
    datasets = []

    for sensor in selected_sensors:
        file_path = DATA_FOLDER / sensor_files[sensor]

        df = pd.read_csv(file_path, sep=";", na_values=["NA"])

        # Add a column identifying the sensor
        df["Sensor"] = sensor

        datasets.append(df)

    # Combine selected datasets
    combined_df = pd.concat(
        datasets,
        ignore_index=True
    )

    

# Combine selected datasets
combined_df = pd.concat(
    datasets,
    ignore_index=True
)

# =====================================
# Section 6.2 - Monitoring Station Filter
# =====================================

st.subheader("Monitoring Station Selection")

selected_station = st.selectbox(
    "Select monitoring station:",
    ["Both", "SUD3", "QDP"]
)

if selected_station == "SUD3":
    combined_df["Reference_NO2"] = combined_df["SUD3_NO2"]

elif selected_station == "QDP":
    combined_df["Reference_NO2"] = combined_df["QDP_NO2"]

else:
    combined_df["Reference_NO2"] = pd.NA

# Display the selected sensor data
st.subheader("Selected Sensor Data")

st.dataframe(
    combined_df,
    use_container_width=True
)

st.write("Total observations:", len(combined_df))

# Display reference measurements
st.subheader("Reference NO₂ Measurements")

if selected_station == "Both":
    st.dataframe(
        combined_df[
            ["Sensor", "date", "SUD3_NO2", "QDP_NO2"]
        ],
        use_container_width=True
    )
else:
    st.dataframe(
        combined_df[
            ["Sensor", "date", "Reference_NO2"]
        ],
        use_container_width=True
    )

    
    st.subheader("Selected Sensor Data")

    # Display the combined dataset
    st.dataframe(
        combined_df,
        use_container_width=True
    )

    # Display total observations
    st.write(
        "Total observations:",
        len(combined_df)
    )

else:
    st.warning("Please select at least one ASE sensor.")

