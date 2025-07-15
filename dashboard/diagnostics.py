# dashboard/diagnostics.py

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.helpers import extract_features


def render_diagnostics(model, test_df: pd.DataFrame) -> None:
    """
    Render global diagnostics for RUL predictions on the TEST set:
    1) Distribution of prediction errors (residuals)
    2) Summary error metrics (bias, RMSE)
    3) Scatter plot of True vs. Predicted RUL for all TEST engines
    """
    st.title("Diagnostics (Test Set)")

    # Introductory guidance
    st.markdown(
        "Explore the model’s performance on the entire test set:\n\n"
        "- **Error Distribution** shows how predictions deviate from true RUL.\n"
        "- **Bias & RMSE** summarise overall accuracy.\n"
        "- **Scatter plot** compares true vs. predicted RUL for each engine."
    )

    # === Prepare full TEST set ===
    full = test_df.copy()
    X = extract_features(full)
    y_true = full["true_RUL"]
    y_pred = model.predict(X)

    df = pd.DataFrame(
        {
            "Engine Unit": full["unit_number"],
            "True RUL (cycles)": y_true,
            "Predicted RUL (cycles)": y_pred,
        }
    )
    # Compute residuals
    df["error"] = df["Predicted RUL (cycles)"] - df["True RUL (cycles)"]

    # 1) Error distribution histogram
    st.subheader("1) Error Distribution")
    hist = (
        alt.Chart(df, title="Prediction Error Distribution on Test Set")
        .mark_bar()
        .encode(
            x=alt.X(
                "error:Q",
                bin=alt.Bin(maxbins=30),
                title="Error e = Predicted − True (cycles)",
            ),
            y=alt.Y("count():Q", title="Number of Engines"),
            tooltip=[alt.Tooltip("count():Q", title="Count of engines")],
        )
        .properties(width=600, height=300)
    )
    st.altair_chart(hist, use_container_width=True)

    # 2) Summary error metrics (aligned left)
    st.subheader("2) Summary Error Metrics")
    bias = df["error"].mean()
    rmse = (df["error"] ** 2).mean() ** 0.5

    col1, col2, _ = st.columns([2, 2, 6])
    with col1:
        st.metric(
            label="Mean Prediction Error (Bias)",
            value=f"{bias:.1f} cycles",
            help="Average of (Predicted − True) over all engines",
        )
    with col2:
        st.metric(
            label="Prediction Error RMSE",
            value=f"{rmse:.1f} cycles",
            help="Root Mean Squared Error over all engines",
        )

    # 3) Scatter True vs. Predicted RUL
    st.subheader("3) True vs. Predicted RUL Scatter")
    base_scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("True RUL (cycles):Q", title="True Remaining Useful Life (cycles)"),
            y=alt.Y(
                "Predicted RUL (cycles):Q",
                title="Predicted Remaining Useful Life (cycles)",
            ),
            color=alt.value("steelblue"),
            tooltip=[
                alt.Tooltip("Engine Unit:N", title="Engine Unit"),
                alt.Tooltip("True RUL (cycles):Q", title="True RUL"),
                alt.Tooltip("Predicted RUL (cycles):Q", title="Predicted RUL"),
            ],
        )
    )

    # Diagonal reference line y = x
    max_val = max(df["True RUL (cycles)"].max(), df["Predicted RUL (cycles)"].max())
    reference_line = (
        alt.Chart(pd.DataFrame({"x": [0, max_val], "y": [0, max_val]}))
        .mark_line(color="red", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )

    layered = (
        alt.layer(base_scatter, reference_line)
        .resolve_scale(x="shared", y="shared")
        .properties(width=600, height=300)
    )
    st.altair_chart(layered, use_container_width=True)
