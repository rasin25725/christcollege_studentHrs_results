import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "student_pass_fail_model2.pkl"
model = joblib.load(model_path)

st.title("Student Pass Predictor")
st.write("Enter the number of hours studied and Attendance to predict the result.")

study_hours = st.number_input("Study hours", min_value=0.0, step=0.5)
attendance = st.number_input("Attendance", min_value=0, step=1)

if st.button("Predict"):
	input_data = pd.DataFrame({"StudyHours": [study_hours],"Attendance":[attendance]})
	prediction = model.predict(input_data)[0]
	probability = model.predict_proba(input_data)[0][int(prediction)]

	if prediction == 1:
		st.success(f"Predicted result: Pass ({probability:.1%} confidence)")
	else:
		st.error(f"Predicted result: Fail ({probability:.1%} confidence)")

