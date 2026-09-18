import gradio as gr
import joblib
import pandas as pd
import os
from pathlib import Path

# Load model
model_path = Path(__file__).parent / "student_pass_fail_modell.pkl"
model = joblib.load(model_path)


def predict_result(study_hours, attendance):

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][int(prediction)]

    if prediction == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return f"Student Result: {result}\nProbability: {probability * 100:.2f}%"


demo = gr.Interface(
    fn=predict_result,
    inputs=[
        gr.Number(
            label="Enter Study Hours",
            minimum=0,
            value=5
        ),
        gr.Number(
            label="Enter Attendance",
            minimum=0,
            value=75
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Student Pass Predictor",
    description="Predict Pass or Fail based on Study Hours and Attendance."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
