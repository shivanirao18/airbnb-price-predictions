Airbnb Price Predictor
An end-to-end ML pipeline that predicts Airbnb listing prices using historical data.
Features:
- Cleans and processes raw Airbnb data
- Trains regression model (Linear Regression)
- Deploys prediction API using Flask
Tech Stack:
- Python (pandas, scikit-learn)
- Flask
- joblib
- Jupyter Notebooks
Setup:
1.Install dependencies
```bash
pip install -r requirements.txt
```
2.Train model
```bash
python model.py
```
3.Run API
```bash
python main.py
```
Send a POST request to `/predict` with:
```json
{
  "features": [feature1, feature2, ...]
}
```
TODO:
- Add Streamlit UI
- Use XGBoost or LightGBM
- Add SHAP visualizations
