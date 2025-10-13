# -*- coding: utf-8 -*-
"""AI Forecasting - UL Example - Predict.ipynb
Original file is located at
    https://colab.research.google.com/drive/1x66N44sQol7zvpu8rEdQcrTK6UBBaa61
"""

!pip install gradio torch scikit-learn pandas requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import base64
import requests
from google.colab import userdata
import gradio as gr
import os
import io
import logging

version='1.0 13 Oct 17:45'

logging.basicConfig(level=logging.INFO)

def git_upload(csv_path,REPO_OWNER,REPO_NAME):

      with open(csv_path, "rb") as f:
          content = f.read()

      encoded_content = base64.b64encode(content).decode()
      url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{csv_path}"
      #GITHUB_TOKEN = userdata.get('GIT_Key2')
      GITHUB_TOKEN=os.environ.get("GITHUB_TOKEN")
      response = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
      if response.status_code == 200:
          sha = response.json()["sha"]
      else:
          sha = None

      payload = {
          "message": "Upload CSV from Colab",
          "content": encoded_content,
      }
      if sha:
          payload["sha"] = sha  # needed for update

      response = requests.put(url, json=payload, headers={"Authorization": f"token {GITHUB_TOKEN}"})

      if response.status_code in [200, 201]:
          print("File uploaded successfully!")
      else:
          print("Error:", response.json())

      logging.info("Using GITHUB_TOKEN: %s", GITHUB_TOKEN[:4] if GITHUB_TOKEN else "None")
      logging.info("PUT payload: %s", payload)
      logging.info("GitHub response code: %s, body: %s", response.status_code, response.text)


def predict_CF(models_bundle, RF_values):
    """
    Perform inference for all (Line_Item, Year) models given ordered RF_values.

    Parameters
    ----------
    models_bundle : dict of {(line_item, year): {'model': model, 'y_scaler': scaler, 'base_value': float}}
    RF_values : list or np.ndarray of ordered list of RF values (must match training input order).

    Returns
    -------
    pd.DataFrame of results with columns [Line_Item, Year, Predicted_CF]
    """

    # Wrap in single-row DataFrame for convenience
    input_df = pd.DataFrame([RF_values])

    results = []

    for (line_item, year), info in models_bundle.items():
        model = info['model']
        y_scaler = info['y_scaler']
        base_value = info['base_value']

        # Prepare tensor
        X_scaled = torch.tensor(input_df.values, dtype=torch.float32)

        # Predict scaled
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_scaled)

        preds_scaled_np = preds_scaled.numpy().flatten()

        # Unscale and add base CF
        preds_delta_unscaled = y_scaler.inverse_transform(preds_scaled_np.reshape(-1, 1)).flatten()
        pred_unscaled = base_value + preds_delta_unscaled

        results.append({
            'Line_Item': line_item,
            'Year': year,
            'CF': pred_unscaled[0],
            'scenario_id': 100,
            'scenario_type': 'Forecast',
            'RF1': RF_values[0],
            'RF2': RF_values[1],
            'RF3': RF_values[2],
            'RF4': RF_values[3],
            'RF5': RF_values[4],
            'RF6': RF_values[5],
            'RF7': RF_values[6]
        })

    return pd.DataFrame(results)

class PredictionNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=16, dropout_prob=0.1):
        super(PredictionNet, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)   # stabilizes training
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # scalar output

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

from google.colab import auth
auth.authenticate_user()

# 1. Identify Models
response = requests.get("https://github.com/chelsee-ai/ai-forecasting/raw/refs/heads/main/models_bundle.pt")
response.raise_for_status()  # ensure download succeeded
models_bundle = torch.load(io.BytesIO(response.content), weights_only=False)

def run_forecast(RF1, RF2, RF3, RF4, RF5, RF6, RF7):
    RF_values = [RF1 / 100, RF2 / 100, RF3, RF4 / 100, RF5 / 100, RF6 / 100, RF7 / 100]
    df_preds = predict_CF(models_bundle, RF_values)
    df_preds["CF"] = df_preds["CF"].round(2)
    # Save locally
    csv_path = "Forecast_Live_UL.csv"
    df_preds.to_csv(csv_path, index=False)

    # Upload to GitHub (optional)
    try:
        git_upload(csv_path, "chelsee-ai", "ai-forecasting")
    except Exception as e:
        print("Upload skipped or failed:", e)
        logging.exception("Git upload failed")

    return df_preds

logo_url = "https://raw.githubusercontent.com/chelsee-ai/ai-forecasting/main/logo.png"

css = """
/* Run button style */
button {background-color: #118DFF !important; color: white !important; border-radius: 6px;}
button:hover {background-color: #0d75d9 !important;}

/* RF input boxes narrower */
.rf-input input {width: 80px !important;}

/* RF labels blue */
.gr-number label, .gr-number span {color: #118DFF !important; font-weight:bold;}

/* Hide Clear and Flag buttons */
button[data-testid="clear-button"], button[data-testid="flag-button"] {display: none !important;}
"""

with gr.Blocks(css=css) as app:
    # Header with logo
    gr.HTML(f"""
    <center>
        <img src="{logo_url}" style="width:90px;margin-bottom:5px;">
        <h1 style="color:#118DFF;">Chelsee AI - Live Forecasting</h1>
        <h3 style="color:#118DFF;">{version}</h3>
    </center>
    """)

    # Row of RF inputs
    with gr.Row():
        RF1 = gr.Number(label="Equity Ret 25 (%)", value=1.0, elem_classes="rf-input")
        RF2 = gr.Number(label="Equity Ret LT (%)", value=-3.0, elem_classes="rf-input")
        RF3 = gr.Number(label="NB Prem 25 ($)", value=10000, elem_classes="rf-input")
        RF4 = gr.Number(label="NB Growth LT (%)", value=0.3, elem_classes="rf-input")
        RF5 = gr.Number(label="Lapse (%)", value=-0.1, elem_classes="rf-input")
        RF6 = gr.Number(label="Short Rate (%)", value=2.0, elem_classes="rf-input")
        RF7 = gr.Number(label="YC Slope (%)", value=0.4, elem_classes="rf-input")

    # Output table
    df_output = gr.Dataframe(label="Forecasted cashflows:")

    # Run button
    run_btn = gr.Button("Run")

    # Connect button
    run_btn.click(run_forecast, inputs=[RF1, RF2, RF3, RF4, RF5, RF6, RF7], outputs=df_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.launch(
        server_name="0.0.0.0",   # must be all interfaces
        server_port=port,        # must match Cloud Run PORT
        share=False,             # Cloud Run provides its own URL
        show_api=False
    )
