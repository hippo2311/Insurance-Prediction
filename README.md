# Medical Insurance Cost Prediction Project

This is a **medical insurance cost prediction project** (not a competition).  
The notebook covers EDA, feature engineering, modeling, tuning, and explainability.

## Objective
Predict `charges` from:
`age, sex, bmi, children, smoker, region`.

## Dataset
File: `data/medical_insurance.csv`  
Size: **1,338 rows × 7 columns**  
Columns: `age, sex, bmi, children, smoker, region, charges`

## What the notebook does (`Mini_Hackathon.ipynb`)
- Data loading & cleaning
- EDA by:
  - Age (distribution, age groups, smoker interaction)
  - BMI (WHO bands, relationship with charges)
  - Sex/Smoker (bar charts, ridgeline)
  - Children & Region
- Feature engineering:
  - Centering + squared terms (`age_c, bmi_c, age_sq, bmi_sq`)
  - Hinge features (`age_h59, bmi_h25/30/35`)
  - BMI class, Age_Group
  - Binary encoding for smoker/sex
  - Interaction features (`smoker×age, smoker×bmi, age×bmi`)
  - KMeans risk cluster features
- Modeling:
  - Linear/regularized, tree/ensemble, MLP
  - Optional XGBoost/LightGBM/CatBoost
  - Metrics: MAE, RMSE, R2
- Tuning & blending:
  - Holdout tuning (no-CV) for selected models
  - Random search over predefined parameter spaces
  - Optional blend-2 / blend-3 weight search on validation
- Explainability:
  - LIME local explanations
  - Global feature importance
  - Reliability & outlier analysis

## Results (test set – extracted from notebook)
**Primary metric: MAE**  
Top models by MAE:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| **HuberRegressor** | **1969.73** | 4523.75 | 0.8596 |
| GradientBoosting | 2483.89 | 4361.76 | 0.8695 |
| RandomForest | 2633.11 | 4556.75 | 0.8575 |
| Lasso | 2646.32 | 4364.49 | 0.8693 |
| LinearRegression | 2647.75 | 4363.98 | 0.8693 |

> Full results are shown in the notebook output (`results` cell).

## Key insights
- **Smoker status is the strongest driver** of high charges.
- **Charges increase with age and BMI**, especially at higher BMI ranges.
- **Region has a weaker effect** compared to smoker/age/BMI.
- **Sex differences are small** relative to smoker impact.
- Some tree models **overfit** (very low train error but worse test error).

## Explainability notes
- **Strongest importance (HuberRegressor, |coef| %):**
  1. `smoker_x_bmi` — 37.90%
  2. `smoker_raw` — 10.79%
  3. `smoker` — 10.79%
  4. `cell_mean_cred_x` — 5.65%
  5. `cell_mean_cred_y` — 5.65%
  6. `risk_cell` — 4.96%
  7. `age` — 4.61%
  8. `age_c` — 4.61%
  9. `smoker_x_age` — 2.21%
  10. `children` — 1.90%
- **Extreme cases:** Two outliers (instances 44 & 197) are severely underestimated  
  (~$23k actual vs ~$4k predicted). LIME shows strong negative contributions from  
  `smoker_x_bmi_h30`, `smoker_x_bmi`, and `smoker=0`, which push predictions down.

## Limitations
- Single train/test split (no cross-validation).
- Tuning uses a holdout validation set; results vary by split.
- LIME is a local approximation and can be unstable for some samples.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Mini_Hackathon.ipynb
```

## Project structure
```
.
├── Mini_Hackathon.ipynb
├── README.md
├── requirements.txt
└── data
    └── medical_insurance.csv
```
