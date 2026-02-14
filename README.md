# Flight Delay Prediction

## Project Overview
This project implements a machine learning pipeline to predict flight departure delays of 15 minutes or more (`DepDel15`) using integrated operational and weather data. By analyzing 105,783 flight records across 10 major U.S. airports, the system identifies high-risk delay windows to support proactive airline operations planning.

### Key Performance Metrics (XGBoost)
*   **ROC-AUC:** 0.735
*   **Recall:** 0.543
*   **PR-AUC:** 0.448

## Data Engineering Pipeline
The system utilizes a modular processing architecture to merge disparate datasets:

1.  **Flight Data Processing:**
    *   Ingested 3 months of Bureau of Transportation Statistics (BTS) data.
    *   Filtered to 10 primary hubs (ATL, EWR, JFK, LAS, LAX, MCO, MIA, ORD, SEA, SFO).
    *   Feature selection reduced 100+ raw columns to 15 high-impact predictors.
2.  **Weather Data Integration:**
    *   Flattened hourly JSON observations into structured features.
    *   Extracted: `tempC`, `windspeedKmph`, `precipMM`, `visibility`, and `weatherCode`.
3.  **Feature Engineering:**
    *   Temporal alignment: Merged datasets on `Airport`, `Date`, and `Departure Hour`.
    *   Engineered `dep_hour` and `DayOfWeek` to capture cyclic delay patterns.

## Key Insights
*   **Propagation Effect:** Delay probability scales from ~10% in the morning to ~30% in the evening.
*   **Non-Linearity:** Weather impacts (wind/visibility) exhibit threshold-based behavior rather than linear correlations.
*   **Interaction Driven:** The strongest predictors are interactions between `Departure Hour` and `Day of Week`.

## Modeling & Evaluation
We compared multiple architectures, prioritizing **Recall** to maximize early-warning capabilities for operational risk mitigation.

### Model Comparison
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Final)** | **0.735** | **0.448** | 0.403 | **0.543** | **0.463** |
| Random Forest | 0.730 | 0.436 | 0.443 | 0.461 | 0.452 |
| Gradient Boosting | 0.730 | 0.440 | 0.461 | 0.404 | 0.431 |
| Logistic Regression | 0.679 | 0.321 | 0.423 | 0.221 | 0.291 |

### Final Model Configuration
*   **Algorithm:** XGBoost
*   **Hyperparameters:** `learning_rate: 0.1`, `max_depth: 4`, `n_estimators: 300`, `subsample: 0.8`.
*   **Interpretability:** SHAP analysis identifies `dep_hour`, `Origin`, and `windgustKmph` as the primary drivers of model decisions.

## Tech Stack
*   **Language:** Python 3.13.5
*   **Data Science:** `pandas`, `numpy`, `scikit-learn`
*   **ML Framework:** `xgboost`
*   **Visualization:** `matplotlib`, `seaborn`, `shap`

## Future Roadmap
*   **Data Expansion:** Incorporate full-year seasonality and aircraft tail-number rotation tracking.
*   **Advanced Features:** Integrate real-time METAR/TAF weather alerts and airport congestion metrics.
*   **Deployment:** Implement temporal cross-validation and an automated early-warning alert system.

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Execution Order
Run the following notebooks sequentially to reproduce the results:
1. `00_data_directory.ipynb`
2. `01_flight_data_preprocessing.ipynb`
3. `02_weather_data_preprocessing.ipynb`
4. `03_merge_flight_weather_data.ipynb`
5. `04_exploratory_data_analysis.ipynb`
6. `05_baseline_modeling_data.ipynb`
7. `06_final_modeling_data.ipynb`