# dashboard/signals.py

import pandas as pd
import streamlit as st

from dashboard.helpers import get_unit_row


def render_signals(test_df: pd.DataFrame, units: list[int]) -> None:
    """
    Display raw sensor and operational setting values for a selected engine.
    """
    st.title("Raw Sensor & Operational Settings")

    # Introductory guidance
    st.markdown(
        "Use this tab to inspect the raw sensor readings and operational settings "
        "for any engine in the test set."
    )

    # Engine selector within the test set
    selected = st.selectbox(
        "Select Engine Unit (Test Set)",
        options=[None] + units,
        format_func=lambda x: "— select an engine —" if x is None else f"Engine {x}",
        key="signals_selector",
        help="Pick an engine from the test set to view its raw sensor and setting values.",
    )
    if selected is None:
        st.info("Please select an engine from the test set to view raw data.")
        return

    # Retrieve the corresponding row
    row = get_unit_row(test_df, selected)

    # Show raw values in an expander for cleanliness
    with st.expander(f"Engine {selected}: Raw Sensor & Settings Data", expanded=True):
        # Drop non‑feature columns and transpose for readability
        df_raw = row.drop(
            columns=["unit_number", "time_in_cycles", "true_RUL"], errors="ignore"
        ).T
        df_raw.columns = [f"Engine {selected}"]

        # Render the table
        st.dataframe(df_raw, use_container_width=True)
