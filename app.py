"""Advanced Data Science: interactive ASE sensor and reference NO₂ dashboard.

Expected repository structure:
    app.py
    requirements.txt
    data/ASE4.csv ... data/ASE13.csv
    data/model_results.csv  (exported from Colab Section 5.3)
"""

# Path provides a platform-independent way to build paths to local files.
from pathlib import Path

# Pandas handles the tabular CSV data and the summary calculations.
import pandas as pd
# Plotly Express creates the interactive line and bar charts.
import plotly.express as px
# Streamlit supplies the dashboard controls, tables, tabs and messages.
import streamlit as st

# ============================================================
# SECTION 6.6 - PAGE SETTINGS AND NAVIGATION
# ============================================================
# Set the browser title, icon and wide layout before drawing page elements.
st.set_page_config(page_title="NO₂ Prediction Dashboard", page_icon="🌍", layout="wide")
st.title("Air Quality NO₂ Prediction Dashboard")
st.caption("Advanced Data Science Project | ASE Sensor Analysis and Machine Learning")

# Build the data directory relative to this script, not the working directory.
DATA_FOLDER = Path(__file__).resolve().parent / "data"
# Map each displayed sensor name to its CSV filename in the data folder.
SENSOR_FILES = {
    "ASE4": "ASE4.csv", "ASE5": "ASE5.csv", "ASE6": "ASE6.csv",
    "ASE7": "ASE7.csv", "ASE8": "ASE8.csv", "ASE10": "ASE10.csv",
    "ASE11": "ASE11.csv", "ASE12": "ASE12.csv", "ASE13": "ASE13.csv",
}
# Map the station choices to the corresponding CSV column names.
REFERENCE_COLUMNS = {"SUD3": "SUD3_NO2", "QDP": "QDP_NO2"}


# Cache CSV reading so changing a filter does not repeatedly reload the same files.
@st.cache_data(show_spinner=False)
# Read one CSV file; caching avoids unnecessary repeated disk reads.
# Add the selected sensor label so rows remain identifiable after combining files.
def load_sensor_file(file_path: str, sensor_name: str) -> pd.DataFrame:
    # The original datasets use semicolon delimiters and 'NA' for missing values.
    frame = pd.read_csv(file_path, sep=";", na_values=["NA"])
    frame["Sensor"] = sensor_name
    return frame


@st.cache_data(show_spinner=False)
# Read the saved Colab evaluation table only when needed.
def load_model_results(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# ============================================================
# SECTION 6.1 - SIDEBAR SENSOR SELECTION AND DATA LOADING
# ============================================================
st.sidebar.title("Dashboard Filters")
st.sidebar.write("Choose the ASE sensors, reference station and recording period.")
# Allow multiple sensors to be selected simultaneously in the sidebar.
selected_sensors = st.sidebar.multiselect(
    "ASE sensors", options=list(SENSOR_FILES), default=["ASE4", "ASE5"]
)
# Choose which reference station columns appear in tables and charts.
selected_station = st.sidebar.selectbox("Reference monitoring station", ["Both", "SUD3", "QDP"])

# Stop early when no sensor has been selected, avoiding empty-data errors.
if not selected_sensors:
    st.info("Select at least one ASE sensor in the sidebar to view the dashboard.")
    st.stop()

# Store each successfully loaded sensor table before combining them.
loaded_frames = []
# Load each selected sensor independently and check that its file exists.
for sensor in selected_sensors:
    path = DATA_FOLDER / SENSOR_FILES[sensor]
    # Show a helpful error if a requested dataset has not been uploaded.
    if not path.is_file():
        st.error(f"Missing dataset: data/{path.name}. Upload this CSV to the repository's data folder.")
        st.stop()
    try:
        # Use the cached CSV reader and attach the current sensor name.
        loaded_frames.append(load_sensor_file(str(path), sensor))
    # Handle file and CSV parsing errors with a readable dashboard message.
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to read {path.name}: {error}")
        st.stop()

# Keep the raw combined dataset separate from derived columns used for charts.
# Stack selected datasets row-wise into one table for analysis.
combined_df = pd.concat(loaded_frames, ignore_index=True)
# These columns are needed by the date filter and NO₂ visualisations.
required_columns = {"date", "ASE_NO2", "SUD3_NO2", "QDP_NO2"}
# Detect absent columns before trying to access them later.
missing_columns = required_columns - set(combined_df.columns)
if missing_columns:
    st.error("Required columns are missing: " + ", ".join(sorted(missing_columns)))
    st.stop()

# The station selector chooses which reference measurements to display.
# It does NOT filter observations by site: the raw CSV location mapping
# would need to be verified before doing that.
# Select one or both reference measurement columns without dropping any rows.
reference_cols = (
    list(REFERENCE_COLUMNS.values()) if selected_station == "Both"
    else [REFERENCE_COLUMNS[selected_station]]
)

# ============================================================
# SECTION 6.5 - DATE RANGE FILTER
# ============================================================
# Convert dates once and keep the original data for quality analysis.
# Create a working copy so conversions do not change the raw quality statistics.
chart_df = combined_df.copy()
# Convert date text to timestamps; invalid entries become missing timestamps.
chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
# Convert plotted values to numbers; nonnumeric entries become missing values.
for col in ["ASE_NO2", *REFERENCE_COLUMNS.values()]:
    chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

# Find the available recording period using only valid timestamps.
valid_dates = chart_df["date"].dropna()
# Avoid showing a date picker with invalid minimum or maximum dates.
if valid_dates.empty:
    st.sidebar.warning("No valid dates found; time-series charts are unavailable.")
    filtered_df = chart_df.iloc[0:0].copy()
else:
    # Determine the earliest and latest dates across the selected sensor files.
    earliest, latest = valid_dates.min().date(), valid_dates.max().date()
    # Let users narrow the displayed graph period using the sidebar.
    chosen_dates = st.sidebar.date_input(
        "Recording period", value=(earliest, latest),
        min_value=earliest, max_value=latest
    )
    # A complete date range contains two values; a partial choice may contain one.
    if isinstance(chosen_dates, (tuple, list)) and len(chosen_dates) == 2:
        start_date, end_date = chosen_dates
        # Only apply the filter when the date range is in chronological order.
        if start_date <= end_date:
            # Keep rows whose dates fall within the selected inclusive range.
            filtered_df = chart_df.loc[
                chart_df["date"].dt.date.between(start_date, end_date)
            ].copy()
        else:
            st.sidebar.warning("Start date must be before the end date.")
            filtered_df = chart_df.iloc[0:0].copy()
    else:
        st.sidebar.info("Select both a start and end date.")
        filtered_df = chart_df.iloc[0:0].copy()

# Tabs provide navigation without creating multiple Streamlit pages.
# Organise the dashboard into four sections without requiring separate pages.
overview_tab, graphs_tab, models_tab, quality_tab = st.tabs([
    "Overview & Data", "Sensor Graphs", "Model Comparison", "Data Quality"
])

# ============================================================
# SECTION 6.2 - OVERVIEW AND REFERENCE MEASUREMENTS
# ============================================================
# Display high-level totals and the original measurements for selected sensors.
with overview_tab:
    st.header("Project Overview")
    st.write(
        "Explore raw ASE sensor signals and NO₂ reference measurements "
        "from the SUD3 and QDP monitoring stations. Compare the regression "
        "models developed in Google Colab."
    )
    # Place three headline metrics side by side.
    a, b, c = st.columns(3)
    a.metric("Selected sensors", len(selected_sensors))
    b.metric("Total observations", f"{len(combined_df):,}")
    c.metric("Observations in date range", f"{len(filtered_df):,}")

    st.subheader("Selected Sensor Data")
    st.dataframe(combined_df, use_container_width=True)

    st.subheader("Reference NO₂ Measurements")
    st.caption("The station choice selects reference columns; it does not filter sensor rows by location.")
    st.dataframe(
        combined_df[["Sensor", "date", *reference_cols]],
        use_container_width=True, hide_index=True
    )

# ============================================================
# SECTION 6.3 - INTERACTIVE TIME-SERIES GRAPHS
# ============================================================
# Show time-series plots only for the chosen date range.
with graphs_tab:
    st.header("Interactive Sensor and Reference Graphs")
    # Explain what should be checked before the dashboard is submitted.
    st.caption(
        "ASE_NO2 may represent an uncalibrated sensor signal rather than NO₂ "
        "concentration. Compare its shape with reference measurements, not "
        "its numerical magnitude, unless units and calibration are confirmed."
    )
    st.write(f"Observations in selected date range: **{len(filtered_df):,}**")
    # Give users the option to hide or show the reference-station chart.
    show_reference = st.checkbox("Show reference station measurements", value=True)

    # Avoid plotting when no valid dated rows match the current selection.
    if filtered_df.empty:
        st.warning("No valid dated measurements are available for this selection.")
    else:
        # Separate charts avoid implying that raw ASE and reference values
        # necessarily share the same units or measurement scale.
        # Exclude missing ASE readings so the plotted line uses valid values.
        ase_graph = filtered_df.dropna(subset=["ASE_NO2"])
        if not ase_graph.empty:
            # Plot each selected ASE sensor as a separate line.
            fig_ase = px.line(
                ase_graph.sort_values("date"), x="date", y="ASE_NO2",
                color="Sensor", title="ASE NO₂ Sensor Signal Over Time",
                labels={"date": "Date and Time", "ASE_NO2": "ASE_NO2 (raw value; units unverified)"}
            )
            st.plotly_chart(fig_ase, use_container_width=True)
        else:
            st.info("No ASE_NO2 readings are available in the selected period.")

        # Build reference-station lines only when the checkbox is enabled.
        if show_reference:
            # Reference readings may be repeated in each sensor file.
            # Remove exact date/value duplicates for each station, without
            # assuming all sensors have the same timestamps.
            # Collect the selected station series before combining them for plotting.
            ref_parts = []
            # Prepare a separate reference series for each selected monitoring station.
            for col in reference_cols:
                # Remove missing and exact duplicate timestamp/value pairs from the reference series.
                ref = filtered_df[["date", col]].dropna().drop_duplicates().copy()
                ref = ref.rename(columns={col: "Reference_NO2"})
                # Add a readable legend label for each reference station.
                ref["Station"] = col.replace("_NO2", "")
                ref_parts.append(ref)
            if ref_parts:
                # Join the prepared station series into one long-form plotting table.
                reference_plot = pd.concat(ref_parts, ignore_index=True)
                if not reference_plot.empty:
                    # Plot the reference measurements separately from raw ASE sensor signals.
                    fig_ref = px.line(
                        reference_plot.sort_values("date"), x="date", y="Reference_NO2",
                        color="Station", title="Reference Station NO₂ Measurements",
                        labels={"date": "Date and Time", "Reference_NO2": "Reference NO₂ (dataset units)"}
                    )
                    st.plotly_chart(fig_ref, use_container_width=True)
                else:
                    st.info("No reference NO₂ measurements are available in this period.")

# ============================================================
# SECTION 6.4 - MACHINE LEARNING MODEL COMPARISON
# ============================================================
# Show saved model evaluation results rather than training models in Streamlit.
with models_tab:
    st.header("Machine Learning Model Comparison")
    st.write("Results exported from the four regression models in Colab Section 5.3.")
    # Locate the CSV exported from the Colab model evaluation section.
    results_path = DATA_FOLDER / "model_results.csv"
    # Show an upload reminder when the results CSV is unavailable.
    if not results_path.is_file():
        st.warning("Upload model_results.csv to the repository's data folder.")
    else:
        try:
            # Load the results table from the repository data directory.
            results_df = load_model_results(str(results_path))
            # Require the model name and three evaluation metrics for comparison.
            expected = {"Model", "MAE", "RMSE", "R2"}
            # Identify missing fields before constructing charts or finding the best model.
            absent = expected - set(results_df.columns)
            if absent:
                st.error("Model results file is missing columns: " + ", ".join(sorted(absent)))
            else:
                # Ensure the evaluation metrics can be compared numerically.
                for metric in ("MAE", "RMSE", "R2"):
                    results_df[metric] = pd.to_numeric(results_df[metric], errors="coerce")
                st.subheader("Model Evaluation Results")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                # Populate the model selector using names actually present in the CSV.
                model_options = results_df["Model"].dropna().astype(str).tolist()
                # Allow users to compare any subset of the available models.
                chosen_models = st.multiselect(
                    "Regression models to compare", model_options, default=model_options
                )
                # Switch the comparison chart between MAE, RMSE and R².
                metric = st.selectbox("Evaluation metric", ["MAE", "RMSE", "R2"])
                # Restrict the plot to chosen models with a valid value for this metric.
                selected_results = results_df[results_df["Model"].isin(chosen_models)].dropna(subset=[metric])
                if not selected_results.empty:
                    # Draw a bar for each selected regression model.
                    fig_models = px.bar(
                        selected_results, x="Model", y=metric, color="Model",
                        title=f"Regression Model Performance: {metric}"
                    )
                    st.plotly_chart(fig_models, use_container_width=True)
                else:
                    st.info("Select models with valid results to display a comparison.")
                # Ignore models without RMSE values when identifying the lowest error.
                valid_rmse = results_df.dropna(subset=["RMSE"])
                if not valid_rmse.empty:
                    # Find the row corresponding to the smallest available RMSE.
                    best = valid_rmse.loc[valid_rmse["RMSE"].idxmin()]
                    st.success(f"Lowest RMSE: {best['Model']} ({best['RMSE']:.4f})")
                st.caption("Lower MAE and RMSE are better; higher R² is generally better.")
        # Handle file and CSV parsing errors with a readable dashboard message.
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Unable to load model results: {error}")

# ============================================================
# SECTION 6.7 - DATA QUALITY AND DESCRIPTIVE STATISTICS
# ============================================================
# Analyse the original loaded rows, independently of the graph date filter.
with quality_tab:
    st.header("Data Quality and Statistics")
    st.write("These statistics use the original selected CSV rows, before date filtering.")
    # Use only source columns and the sensor identifier. No artificial
    # Reference_NO2 column is added, so it cannot inflate missing counts.
    # Count empty cells separately for each original dataset column.
    missing_counts = combined_df.isna().sum()
    # Express missing cells as a percentage of the total selected rows.
    missing_pct = combined_df.isna().mean().mul(100)
    total_missing = int(missing_counts.sum())
    # Place three headline metrics side by side.
    a, b, c = st.columns(3)
    a.metric("Total observations", f"{len(combined_df):,}")
    b.metric("Selected sensors", len(selected_sensors))
    c.metric("Missing cells", f"{total_missing:,}")

    st.subheader("Missing Value Analysis")
    # Build a readable table with counts and percentages for each column.
    missing_table = pd.DataFrame({
        "Column": combined_df.columns,
        "Missing Values": missing_counts.to_numpy(),
        "Missing (%)": missing_pct.to_numpy()
    })
    st.dataframe(missing_table, use_container_width=True, hide_index=True)
    # Visualise which columns contain the greatest share of missing values.
    fig_missing = px.bar(
        missing_table, x="Column", y="Missing (%)",
        title="Percentage of Missing Values by Column"
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    st.subheader("Descriptive Statistics")
    # Some numeric columns can be read as text if they contain mixed entries.
    # Describe columns that Pandas has identified as numeric in the raw CSVs.
    # Keep numeric columns for mean, standard deviation and other statistics.
    numeric_data = combined_df.select_dtypes(include="number")
    # Handle datasets where no numeric columns are recognised.
    if numeric_data.empty:
        st.info("No numeric columns were detected for descriptive statistics.")
    else:
        st.dataframe(numeric_data.describe(), use_container_width=True)

    st.subheader("Missing Values by Sensor")
    # Count missing cells per sensor without relying on newer Pandas options.
    # Count missing entries across all columns for each individual row.
    sensor_missing = combined_df.isna().sum(axis=1)
    # Group row-level missing counts by sensor for a per-sensor summary.
    by_sensor = pd.DataFrame({
        "Sensor": combined_df["Sensor"],
        "Missing Cells": sensor_missing
    }).groupby("Sensor", as_index=False).agg(
        Observations=("Missing Cells", "size"),
        **{"Missing Cells": ("Missing Cells", "sum")}
    )
    st.dataframe(by_sensor, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 6.8 - FINAL TESTING NOTES
# ============================================================
st.divider()
# Explain what should be checked before the dashboard is submitted.
st.caption(
    "Validation: check all nine CSVs, sensor/station/date filters, chart labels, "
    "missing-value counts and model_results.csv against the Colab notebook. "
    "This dashboard displays saved model results; it does not retrain models."
)
