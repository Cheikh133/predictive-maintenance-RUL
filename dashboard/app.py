# dashboard/app.py

import sys
from pathlib import Path

# Ensure project root on PYTHONPATH so we can import src/ and dashboard/*
proj_root = Path(__file__).parent.parent
sys.path.append(str(proj_root))

import streamlit as st

from dashboard.diagnostics import render_diagnostics
from dashboard.explainability import render_explainability
from dashboard.helpers import get_unit_ids, load_rul_model, load_test_data
from dashboard.overview import render_overview
from dashboard.signals import render_signals


def main() -> None:
    """Entry point: set up tabs and dispatch to each page renderer."""
    st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")
    st.title("Turbofan Engine Remaining Useful Life")

    # Load model and data once
    model = load_rul_model()
    test_df = load_test_data()
    units = get_unit_ids(test_df)

    # Tabs: Overview, Signals, Diagnostics, Explainability
    tabs = st.tabs(["Overview", "Signals", "Diagnostics", "Explainability"])

    # Overview tab: selection + prediction + importances
    with tabs[0]:
        render_overview(model)

    # Signals tab: sensor time-series
    with tabs[1]:
        render_signals(test_df, units)

    # Diagnostics tab: global performance scatter + error histo
    with tabs[2]:
        render_diagnostics(model, test_df)

    # Explainability tab: SHAP plots
    with tabs[3]:
        # we pass None; the function will prompt the user to pick an engine internally
        render_explainability(model, test_df, None)


if __name__ == "__main__":
    main()
