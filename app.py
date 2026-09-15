from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.recommender import MatrixFactorizationRecommender


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎓",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_interactions.csv"
METRICS_PATH = PROJECT_ROOT / "model_metrics.json"


# ---------------------------------------------------------
# LOAD DATA AND TRAIN PRIVACY-SAFE DEMO MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_demo_model():
    interactions = pd.read_csv(DATA_PATH)

    model = MatrixFactorizationRecommender(
        factors=5,
        epochs=20,
        random_state=42
    )

    model.fit(interactions)

    return interactions, model


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


interactions, model = load_demo_model()
metrics = load_metrics()


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.title("AI-Powered Career Recommendation System")

st.write(
    "A privacy-safe demonstration of personalized career "
    "recommendations using Matrix Factorization."
)

st.info(
    "This public application uses completely synthetic student "
    "interactions. Original student data is not included."
)


# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.subheader("Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Matrix Factorization RMSE",
    f"{metrics['matrix_factorization_rmse']:.4f}"
)

metric_col2.metric(
    "Hit Rate@5",
    f"{metrics['hit_rate_at_5']:.2%}"
)

metric_col3.metric(
    "NDCG@5",
    f"{metrics['ndcg_at_5']:.4f}"
)


# ---------------------------------------------------------
# STUDENT INPUT
# ---------------------------------------------------------

st.subheader("Generate Recommendations")

student_ids = sorted(
    interactions["student_id"].astype(str).unique().tolist()
)

selected_student = st.selectbox(
    "Select a synthetic student",
    student_ids
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# ---------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ---------------------------------------------------------

if st.button("Recommend Careers", type="primary"):

    recommendations = model.recommend(
        selected_student,
        top_n=top_n
    )

    recommendations["career"] = (
        "Career "
        + recommendations["career_id"].astype(str)
    )

    recommendations = recommendations.rename(
        columns={
            "career_id": "Career ID",
            "predicted_score": "Predicted Score"
        }
    )

    recommendations["Predicted Score"] = (
        recommendations["Predicted Score"].round(4)
    )

    st.success(
        f"Top {len(recommendations)} recommendations "
        f"generated for {selected_student}."
    )

    st.dataframe(
        recommendations[
            ["career", "Career ID", "Predicted Score"]
        ],
        use_container_width=True,
        hide_index=True
    )

    chart_data = recommendations.set_index(
        "career"
    )[["Predicted Score"]]

    st.bar_chart(chart_data)


# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

st.divider()

st.subheader("How the System Works")

st.markdown(
    """
    1. Student–career interaction scores are prepared.
    2. Matrix Factorization learns student and career patterns.
    3. Careers already observed by the student are excluded.
    4. The highest predicted unseen careers are recommended.
    """
)

st.caption(
    "Recommendations support career exploration and do not replace "
    "professional career counselling."
)
