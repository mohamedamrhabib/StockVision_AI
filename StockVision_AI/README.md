# StockVision AI

A Streamlit web app for an academic stock-market trend classification project.

## Run locally

1. Install Python 3.11 (recommended).
2. Open a terminal in this folder.
3. Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run:

   ```bash
   streamlit run app.py
   ```

## Project structure

```text
StockVision_AI/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── stock_model.keras
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── history.pkl
    ├── evaluation.npz
    └── Final_Stock_Data_Macro.csv
```

## Deployment

Push this folder to a GitHub repository, then create an app in Streamlit Community Cloud and select `app.py` as the entry point. Keep the `artifacts` folder in the repository.

## Important limitations

- The supplied dataset ends on 6 September 2022. The latest prediction is based on the final available historical observation, not live market data.
- The output is a next-step trend class (Up/Down), not a predicted stock price.
- EPU and GPR are missing in the supplied CSV and are filled with zero by the original preprocessing pipeline.
- This app is for educational/research demonstration, not financial advice.
