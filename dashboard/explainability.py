# dashboard/explainability.py

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from dashboard.helpers import (
    extract_features,
    get_unit_ids,
    get_unit_row,
    load_test_data,
)
from src.model_utils import load_model


def render_explainability(model, test_df: pd.DataFrame, _: None) -> None:
    """
    Render SHAP-based explainability:
    1) Global summary: bar plot of mean|SHAP| per feature
    2) Local explanation: waterfall plot for a selected engine
    """
    st.subheader("Model Explainability with SHAP")

    # --- 1) Global SHAP summary
    st.markdown(
        "**Global Feature Importance (mean |SHAP|)**  \n"
        "Shows which features on average have the largest impact on model predictions."
    )
    # Compute SHAP values
    X = extract_features(test_df)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Render bar plot of mean absolute SHAP values
    plt.figure(figsize=(4, 2))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("Mean |SHAP| per Feature", fontsize=12, pad=10)
    fig_summary = plt.gcf()
    st.pyplot(fig_summary, bbox_inches="tight")
    plt.clf()

    # --- 2) Local explanation for a selected engine
    st.markdown(
        "**Local Explanation for an Engine**  \n"
        "Waterfall plot showing how individual features push the prediction up or down."
    )
    unit_ids = get_unit_ids(test_df)
    selected = st.selectbox(
        "Select Engine for Local Explanation",
        options=[None] + unit_ids,
        format_func=lambda x: "— select an engine —" if x is None else f"Engine {x}",
        help="Pick one engine unit to see its individual SHAP contribution breakdown.",
        key="shap_local_engine",
    )
    if selected is None:
        st.info("Please select an engine to see the local SHAP explanation.")
        return

    # Extract the single-row data and compute SHAP
    row = get_unit_row(test_df, selected)
    X_row = extract_features(row)
    sv_row = explainer(X_row)

    # Render waterfall plot
    plt.figure(figsize=(4, 2))
    shap.plots.waterfall(sv_row[0], show=False)
    plt.title(f"SHAP Waterfall for Engine {selected}", fontsize=12, pad=10)
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall, bbox_inches="tight")
    plt.clf()

    # Provide a short caption
    st.caption(
        "Bars extending to the right indicate features that pushed the predicted RUL higher; bars extending to the left indicate features that pushed it lower."
    )
