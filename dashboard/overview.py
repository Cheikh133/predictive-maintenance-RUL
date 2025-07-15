# dashboard/overview.py

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dashboard.helpers import (
    extract_features,
    get_unit_ids,
    get_unit_row,
    load_test_data,
)


def render_overview(model) -> None:
    """
    Render the Overview page on the TEST dataset:
    1) Global test set metrics (MAE, RMSE)
    2) Engine selector
    3) Per-engine key metrics (cycle, predicted vs. true RUL)
    4) Engine life stage gauge
    """
    st.title("Overview (Test Set)")

    st.markdown(
        """
        This page shows **global performance** on the entire test set followed by **per-engine diagnostics**.
        Use the selector below to inspect an individual engine’s metrics and its life stage.
        """
    )

    # --- 1) Global test set performance
    test_df = load_test_data()
    X_full = extract_features(test_df)
    y_true_full = test_df["true_RUL"]
    y_pred_full = model.predict(X_full)

    mae = mean_absolute_error(y_true_full, y_pred_full)
    rmse = np.sqrt(mean_squared_error(y_true_full, y_pred_full))

    col_global1, col_global2, _ = st.columns([2, 2, 6])
    col_global1.metric(
        label="Overall Test MAE",
        value=f"{mae:.1f} cycles",
        help="Mean Absolute Error across all test engines",
    )
    col_global2.metric(
        label="Overall Test RMSE",
        value=f"{rmse:.1f} cycles",
        help="Root Mean Squared Error across all test engines",
    )

    st.markdown("---")

    # --- 2) Engine selector
    units = get_unit_ids(test_df)
    selected = st.selectbox(
        "Select Engine Unit (Test Set)",
        options=[None] + units,
        format_func=lambda x: "— select an engine —" if x is None else f"Engine {x}",
        key="overview_selector",
        help="Pick an engine to view its individual metrics",
    )
    if selected is None:
        st.info("Please select an engine to see its overview.")
        return

    # --- 3) Per-engine key metrics
    row = get_unit_row(test_df, selected)
    features = extract_features(row)

    col1, col2, col3 = st.columns(3)
    cycle = int(row["time_in_cycles"].iloc[0])
    col1.metric(
        label="Cycle at Measurement",
        value=cycle,
        help="Operational cycle number when this data point was recorded",
    )

    pred = model.predict(features)[0]
    col2.metric(
        label="Predicted RUL (cycles)",
        value=f"{pred:.1f}",
        help="Model’s predicted Remaining Useful Life at this cycle",
    )

    if "true_RUL" in row.columns:
        true = row["true_RUL"].iloc[0]
        col3.metric(
            label="True RUL (cycles)",
            value=f"{true:.1f}",
            help="Ground‑truth Remaining Useful Life from test labels",
        )

        st.markdown("---")

    # --- 4) Engine life stage gauge (based on true RUL)
    st.subheader("Engine Life Stage")

    # Cycles already operated
    cycle = int(row["time_in_cycles"].iloc[0])
    # True Remaining Useful Life
    true_rul = row["true_RUL"].iloc[0]
    # Estimate total life = operated + remaining
    total_life = cycle + true_rul
    # Life stage ratio
    life_stage = cycle / total_life  # between 0 and 1

    # Reading note
    st.markdown(
        "_Note: This gauge shows the fraction of the engine’s lifespan already consumed, "
        "computed as_  \n"
        "`operated_cycles / (operated_cycles + true_RUL)`."
    )

    st.metric(
        label="Life Stage",
        value=f"{life_stage:.1%}",
        help="Proportion of engine’s life already consumed",
    )
    st.progress(life_stage)
