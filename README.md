# Predictive Analytics Using Historical Data

A beginner-friendly Streamlit dashboard that cleans historical date/value data, trains a Linear Regression model, evaluates it on a chronological test set, and forecasts future values.

## Features
- Upload a CSV file or use built-in sample data.
- Select the date/time and numeric target columns.
- Remove invalid/missing rows and aggregate duplicate dates.
- Evaluate predictions with MAE, RMSE, and R².
- Visualize history, test predictions, and future forecasts.
- Download forecast results as CSV.

## CSV format
Your CSV should contain a date/time column and at least one numeric column, for example:

```csv
Date,Sales
2024-01-01,120
2024-02-01,135
2024-03-01,142
```

Provide at least six valid date/value rows. Forecast quality depends on the amount and quality of historical data. The included Linear Regression model is a baseline and does not capture complex seasonal patterns.

## Run on Windows
1. Install Python 3.11 or 3.12 and Git.
2. Open this project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
5. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
8. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Push to GitHub
Create a new empty GitHub repository named `predictive-analytics-historical-data`. In the terminal, from this project folder, run:

```powershell
git init
git add app.py requirements.txt README.md sample_data.csv .gitignore
git commit -m "Build predictive analytics dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/predictive-analytics-historical-data.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to authenticate, complete the browser sign-in prompt. If the repository already has a remote, inspect it with `git remote -v` before adding another.

## Deploy with Streamlit Community Cloud
1. Push the project to GitHub first.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app** and choose your repository, branch `main`, and main file path `app.py`.
4. Click **Deploy** and wait for the build to finish.
5. Open the public app URL and test uploading a CSV.

## Files
- `app.py` — complete dashboard and machine-learning workflow
- `requirements.txt` — Python packages required for deployment
- `sample_data.csv` — example historical data
- `README.md` — project setup and deployment guide
- `.gitignore` — excludes local environment and generated files
