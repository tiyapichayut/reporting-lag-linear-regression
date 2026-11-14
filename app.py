from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "reporting_lag_model.pkl"


COLLISION_LABELS = {
    "พลิกคว่ำ/ตกถนนในทางตรง": "Straight-road rollover or run-off-road crash",
    "ชนท้าย": "Rear-end collision",
    "พลิกคว่ำ/ตกถนนในทางโค้ง": "Rollover or run-off-road crash on a curve",
    "ชนสิ่งกีดขวาง (บนผิวจราจร)": "Collision with an object on the roadway",
    "ชนในทิศทางตรงกันข้าม (ไม่ใช่การแซง)": "Opposing-direction collision (no overtaking)",
    "อื่นๆ": "Other",
    "ชนคนเดินเท้า": "Pedestrian collision",
    "ชนเป็นมุมบริเวณทางแยก": "Angle collision at an intersection",
    "ชนด้านข้าง": "Sideswipe collision",
    "เลี้ยว/ถอยชน": "Collision while turning or reversing",
}


def collision_label(value):
    return COLLISION_LABELS.get(value.strip(), value)


@st.cache_resource
def load_model(path, modified_at):
    # The modification time refreshes the cache after retraining.
    return joblib.load(path)


st.set_page_config(page_title="Road Accident Reporting Lag")
st.title("Road Accident Reporting Lag")

if not MODEL_PATH.is_file():
    st.error("Run python model.py to create reporting_lag_model.pkl beside app.py.")
    st.stop()

model = load_model(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
features = list(model.feature_names_in_)

encoder = model.named_steps["processor"].named_transformers_["categorical"]
category_options = dict(zip(encoder.feature_names_in_, encoder.categories_))

with st.form("crash_details"):
    accident_date = st.date_input("Accident date")
    inputs = {
        "province": st.selectbox("Province", sorted(category_options["province"])),
        "collision_type": st.selectbox(
            "Collision type",
            sorted(category_options["collision_type"], key=collision_label),
            format_func=collision_label,
        ),
        "total_parties": st.number_input("Total parties involved", min_value=1, step=1),
    }

    if "dist_to_province_center_km" in features:
        inputs["dist_to_province_center_km"] = st.number_input(
            "Distance to province center (km)", min_value=0.0, step=0.1
        )
    if "total_injuries" in features:
        inputs["total_injuries"] = st.number_input("Total injuries", min_value=0, step=1)
    if "fatalities" in features:
        inputs["fatalities"] = st.number_input("Fatalities", min_value=0, step=1)

    submitted = st.form_submit_button("Predict reporting lag")

if submitted:
    crash = pd.DataFrame([inputs], columns=features)
    predicted_days = max(0.0, float(model.predict(crash)[0]))
    # Round to the nearest day for the date-only estimate.
    report_date = accident_date + timedelta(days=int(predicted_days + 0.5))
    st.metric("Estimated reporting lag", f"{predicted_days:,.1f} days")
    st.metric("Estimated report date", report_date.strftime("%d %b %Y"))