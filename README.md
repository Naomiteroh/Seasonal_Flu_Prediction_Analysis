# Seasonal Flu Vaccine Prediction
## A Machine Learning Classification Project

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Kenya experiences two flu seasons annually: March to May 
and October to December. Despite the availability of seasonal 
flu vaccines, vaccination rates remain consistently low 
particularly among younger and lower income populations. 
This represents a preventable public health burden that 
costs lives every year.

This project applies machine learning to predict whether 
an individual will receive the seasonal flu vaccine based 
on their background, opinions and health behaviors. While 
the data originates from the National 2009 H1N1 Flu Survey 
conducted in the United States, the methodology and findings 
are adapted for the **Kenyan public health context**.

The analysis follows the full CRISP-DM data science lifecycle: 
from business understanding and data exploration to model 
building, evaluation and actionable stakeholder recommendations.

---

## Business and Data Understanding

### The Stakeholders

| Stakeholder | Level | How They Use This Project |

| **County Director of Health (CDoH)** | Community | Identifies which demographic and behavioral profiles are least likely to vaccinate, informing WHERE and WHO to target for community outreach every flu season |

| **CEO, County Referral Hospital** | Clinical | Identifies which patient characteristics predict low vaccination uptake, informing which patients clinicians should proactively engage during consultations |

### Business Problem
This project answers four key questions:

- Who is least likely to get the seasonal flu vaccine and why?
- Which behavioral, opinion and demographic factors predict vaccination?
- How accurately can we predict vaccination behavior before the flu season begins?
- How can these insights be translated into effective community and clinical interventions?

### Dataset
The dataset comes from the **National 2009 H1N1 Flu Survey** 
collected by the United States government. 26,707 respondents 
answered questions about their backgrounds, opinions and health 
behaviors alongside whether they received flu vaccines.

| File | Description | Used |

| `training_set_features.csv` | 35 features for 26,707 respondents |  Yes |

| `training_set_labels.csv` | Target variables | Yes — seasonal_vaccine only |

| `test_set_features.csv` | Unseen respondent features |  No — competition only |

| `submission_format.csv` | Competition submission template |  No — not competing |

**Target Variable:** `seasonal_vaccine` 
(binary: 0 = not vaccinated, 1 = vaccinated)

**Class Balance:** 53% not vaccinated / 47% vaccinated — 
a balanced dataset requiring no resampling techniques

**Data Source:** [DrivenData — Flu Shot Learning](https://www.drivendata.org/competitions/66/flu-shot-learning/)

**Data Limitations:**
- Dataset originates from the US 2009 H1N1 Flu Survey and 
  may not perfectly represent the Kenyan population
- Survey data is self-reported and may contain response bias
- Dataset skews older, white and educated
- Race column was excluded as it is not transferable to 
  the Kenyan context where the population is predominantly 
  Black African

---

## Key Findings

### Exploratory Data Analysis Insights

Initial exploration revealed a balanced class distribution 
of 53/47 meaning no resampling techniques were required. 
We explored how different features relate to seasonal 
vaccination uptake. Key insights include:

**Opinion features are the strongest predictors:**
Perceived seasonal flu risk (r=0.390), doctor recommendation 
(r=0.369) and belief in vaccine effectiveness (r=0.362) 
are the three strongest predictors — stronger than any 
demographic or behavioral feature.

**Age is the strongest demographic predictor:**
65+ years: 67.4% vaccinated vs 18-34 years: only 28.5% — 
almost a 40 percentage point difference. Younger populations 
are significantly underserved.

**Uncertainty blocks vaccination more than worry:**
People who say "Don't Know" about vaccine side effects have 
the lowest vaccination rate at 18.1% — lower even than 
people who are very worried. Information gaps are a bigger 
barrier than negative attitudes.

**Healthcare access drives uptake:**
Health insurance (r=0.201) and chronic medical conditions 
(r=0.170) are strong predictors showing that access to 
healthcare is a key driver of vaccination behavior.

**Income predicts vaccination:**
Below poverty: 36.3% vaccinated vs above $75,000: 49.7% — 
socioeconomic barriers to vaccination are real and significant.

**Doctor recommendation is critical:**
Unemployed respondents show the lowest vaccination rate 
at 30.2% — likely reflecting lack of healthcare access 
and doctor contact.

---

## Data Preparation

Before modeling we performed several preprocessing steps 
to ensure data quality and prevent data leakage:

**Feature Selection:** Dropped 14 irrelevant features:
- H1N1 specific features — out of scope for seasonal vaccine prediction
- High missing value columns — employment_industry (49.91%) 
  and employment_occupation (50.44%)
- Negligible predictors — behavioral_antiviral_meds (r=0.006)
- Not transferable to Kenya — race column excluded

**Train Test Split:** 80/20 stratified split performed 
BEFORE any transformations to prevent data leakage. 
Stratification confirmed equal class balance in both sets.

**Preprocessing Pipeline:**
- Numeric features: Median imputation → StandardScaler
- Categorical features: Most frequent imputation → OneHotEncoder
- All transformers fitted on training data only and applied 
  to test set using training statistics

**Why this order matters:** Fitting transformers on the full 
dataset before splitting would allow test set information to 
influence training — a form of data leakage that produces 
overly optimistic performance estimates.

---

## Modeling

We adopted an iterative modeling strategy beginning with a 
simple interpretable baseline and progressively introducing 
more complex models. All models used scikit-learn pipelines 
combining preprocessing and classification in one workflow.

Because the dataset is balanced (53/47) we evaluated models 
primarily using **ROC AUC** — a metric that measures the 
model's ability to distinguish between vaccinated and 
unvaccinated individuals across all classification thresholds.

### Baseline Model — Logistic Regression
Logistic Regression was used as a starting point due to its 
interpretability and suitability for binary classification.

**Performance on Test Set:**
- Accuracy: 77.56%
- Precision: 77.27%
- Recall: 73.38%
- F1 Score: 75.27%
- **ROC AUC: 0.8506**

The model performed well as a baseline but uses a linear 
decision boundary which may miss non-linear relationships 
in the data.

### Tuned Models — Ridge and Lasso Regularization
We tuned the baseline with L2 (Ridge) and L1 (Lasso) 
regularization using GridSearchCV with 5-fold cross validation.

- **Ridge** shrinks all coefficients towards zero reducing 
  overfitting while keeping all features
- **Lasso** shrinks some coefficients to exactly zero 
  performing automatic feature selection

Both achieved **ROC AUC: 0.8506** — confirming the baseline 
was not significantly overfitting.

### Nonparametric Model — Decision Tree
A Decision Tree was introduced to capture non-linear 
relationships. max_depth, min_samples_split and 
min_samples_leaf were tuned using GridSearchCV.

**Performance on Test Set:**
- Accuracy: 76.54%
- **ROC AUC: 0.8356**

The Decision Tree underperformed compared to logistic 
regression on ROC AUC suggesting linear relationships 
dominate in this dataset.

### Ensemble Model — Random Forest
A Random Forest was implemented combining 100-200 decision 
trees using bagging to reduce overfitting and improve 
generalization. n_estimators, max_depth, min_samples_split 
and min_samples_leaf were tuned using GridSearchCV.

**Performance on Test Set:**
- Accuracy: 77.72%
- Precision: 78.51%
- Recall: 71.81%
- F1 Score: 75.01%
- **ROC AUC: 0.8518** ← Best model

---

## Evaluation

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |

| Logistic Regression Baseline | 77.56% | 77.27% | 73.38% | 75.27% | 0.8506 |

| Logistic Regression + Ridge | 77.50% | 77.21% | 73.30% | 75.21% | 0.8506 |

| Logistic Regression + Lasso | 77.56% | 77.27% | 73.38% | 75.27% | 0.8506 |

| Decision Tree | 76.54% | 77.37% | 70.12% | 73.57% | 0.8356 |

| **Random Forest** | **77.72%** | **78.51%** | **71.81%** | **75.01%** | **0.8518** |

### Final Model — Random Forest

Random Forest was selected as the final model based on its 
highest ROC AUC score of 0.8518. However we acknowledge 
an important trade-off:

| Factor | Random Forest | Logistic Regression |

| ROC AUC | 0.8518  | 0.8506 |

| Runtime | Slow  | Fast  |

| Explainability | Low  | High  |

| Parsimony | Low  | High  |

| Ease of Deployment | Complex  | Simple  |

The ROC AUC difference is only **0.0012** — practically 
negligible. In a resource constrained Kenyan clinical 
setting, **Logistic Regression with Lasso regularization 
would be a strong alternative** given its equal performance, 
superior explainability and ease of deployment.

### Top 10 Most Important Features

Based on Random Forest feature importance scores:

1. opinion_seas_risk (r=0.390)
2. doctor_recc_seasonal (r=0.369)
3. opinion_seas_vacc_effective (r=0.362)
4. health_insurance (r=0.201)
5. chronic_med_condition (r=0.170)
6. health_worker (r=0.127)
7. behavioral_touch_face (r=0.120)
8. behavioral_wash_hands (r=0.112)
9. opinion_seas_sick_from_vacc (-0.062)
10. household_children (-0.115)

---

## Strategic Recommendations

### For County Director of Health

**Priority 1 — Target Low Risk Perception Communities:**
Opinion about seasonal flu risk is the single strongest 
predictor. Deploy risk education campaigns before each 
flu season using community health workers.

**Priority 2 — Focus on Younger Populations:**
18-34 year olds show only 28.5% vaccination — the lowest 
of any age group. Partner with universities and workplaces. 
Consider mobile vaccination units at youth frequented locations.

**Priority 3 — Remove Financial Barriers:**
Below poverty respondents show only 36.3% vaccination. 
Provide free seasonal flu vaccination at all public 
health facilities and target unemployed populations.

**Priority 4 — Counter Vaccine Misinformation:**
Belief in vaccine effectiveness (r=0.362) is a top predictor. 
Deploy public education campaigns using trusted community 
leaders as vaccine champions.

### For County Referral Hospital CEO

**Priority 1 — Mandate Doctor Recommendations:**
Doctor recommendation (r=0.369) is the second strongest 
predictor. Implement a clinical protocol requiring all 
clinicians to recommend seasonal flu vaccination during 
every consultation.

**Priority 2 — Target Uninsured Patients:**
Health insurance (r=0.201) is a strong predictor. Offer 
free seasonal flu vaccination regardless of insurance status. 
Flag uninsured patients for proactive outreach before each 
flu season.

**Priority 3 — Prioritize Chronic Condition Patients:**
Integrate seasonal flu vaccination into all chronic disease 
management protocols.

**Priority 4 — Address Side Effect Uncertainty:**
Patients who say "Don't Know" about side effects have the 
lowest vaccination rate at 18.1%. Train clinicians to 
proactively address uncertainty not just worry.

---

## Limitations

**Data Limitations:**
- Dataset originates from the US 2009 H1N1 Flu Survey — 
  may not perfectly represent the Kenyan population
- Self-reported survey data may contain response bias
- Dataset skews older, white and educated limiting 
  generalizability across all demographic groups
- Health insurance has 45.96% missing values which may 
  have introduced imputation bias

**Model Limitations:**
- Historical 2009 data — vaccination behavior may have 
  changed significantly since then
- Random Forest is less interpretable than Logistic 
  Regression making it harder to explain individual 
  predictions to non-technical stakeholders
- Model fairness across demographic subgroups was not tested

**Scope Limitations:**
- Does not account for supply side factors like vaccine 
  availability and cost which are particularly relevant 
  in Kenya
- Geographic variables were US-specific and had to be 
  dropped — a Kenya-specific model would benefit from 
  county level geographic data

---

## Next Steps

- Collect Kenya-specific vaccination survey data to retrain 
  and validate the model in the local context
- Test model fairness across demographic subgroups
- Deploy the model as a scoring tool for use by health 
  workers before each flu season
- Pilot community outreach targeting identified low 
  vaccination groups before the next flu season
- Implement doctor recommendation clinical protocol in 
  county referral hospitals
- Explore Gradient Boosting and Stacking ensemble methods

---

## Repository Structure
```
Seasonal_Flu_Prediction_Analysis/
│
├── Data/
│   ├── Raw_data/               ← Original unmodified datasets
│   │   ├── training_set_features.csv
│   │   ├── training_set_labels.csv
│   │   ├── test_set_features.csv
│   │   └── submission_format.csv
│   │
│   └── Cleaned_data/           ← Cleaned and processed datasets
│       ├── cleaned_data.csv
│       └── processed_data.csv
│
├── Notebooks/
│   ├── 01_Data_cleaning.ipynb  ← Business understanding, data 
│   │                              understanding, cleaning
│   ├── 02_Analysis.ipynb       ← EDA, visualizations, feature
│   │                              selection decisions  
│   └── 03_Final_notebook.ipynb ← Complete ML pipeline, modeling,
│                                  evaluation, recommendations
│
├── Images/                     ← All project visualizations
│
├── Presentation/               ← Non technical presentation
│   └── Seasonal_flu_prediction_analysis_presentation.pdf
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Notebooks

Run the notebooks in the following order:

| Order | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | [01_Data_cleaning.ipynb](Notebooks/01_Data_cleaning.ipynb) | Business understanding, data understanding, duplicate removal, missing values visualization | `cleaned_data.csv` |
| 2 | [02_Analysis.ipynb](Notebooks/02_Analysis.ipynb) | EDA, feature distributions, outlier detection, feature vs target analysis, correlation analysis, feature selection | Visualizations and feature selection decisions |
| 3 | [03_Final_notebook.ipynb](Notebooks/03_Final_notebook.ipynb) | Complete ML pipeline, modeling, evaluation, recommendations | `processed_data.csv`, trained models, recommendations |

---

## How to Reproduce This Analysis

**1. Clone the repository**
```bash
git clone https://github.com/Naomiteroh/Seasonal_Flu_Prediction_Analysis.git
cd Seasonal_Flu_Prediction_Analysis
```

**2. Create a virtual environment**
```bash
python -m venv venv
```

**3. Activate environment**
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Launch Jupyter Notebook**
```bash
jupyter notebook
```

Open notebooks in order:
`01_Data_cleaning` → `02_Analysis` → `03_Final_notebook`

---

## Requirements

Key libraries used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- jupyter

All dependencies and versions are listed in `requirements.txt`.

---

## Non Technical Presentation

A stakeholder focused summary of findings and business 
recommendations is available in the `Presentation/` folder.

[📊 View Presentation](https://github.com/Naomiteroh/Seasonal_Flu_Prediction_Analysis/blob/main/Presentation/Seasonal_flu_prediction_analysis_presentation.pdf)

---

## Citations and References

- **Dataset:** DrivenData. (2020). Flu Shot Learning: Predict 
  H1N1 and Seasonal Flu Vaccines. 
  https://www.drivendata.org/competitions/66/flu-shot-learning/

- **Benchmark Walkthrough:** DrivenData. (2020). Flu Shot 
  Learning Benchmark Walkthrough. Concepts referenced include 
  train-test split with stratification and ROC curve plotting.
  https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/

- **Framework:** Chapman et al. (2000). CRISP-DM 1.0 
  Step-by-step data mining guide.



## Conclusion

This project demonstrates how machine learning can be applied 
to predict seasonal flu vaccination behavior and generate 
actionable public health interventions. By identifying 
high-risk populations before the flu season begins, the 
County Director of Health and County Referral Hospital CEO 
can implement targeted strategies that reduce preventable 
illness and improve vaccination coverage.

Among the evaluated models, **Random Forest achieved the 
highest ROC AUC of 0.8518**. However the negligible 
performance difference of 0.0012 over Logistic Regression 
means that in a resource constrained Kenyan clinical setting, 
Logistic Regression with Lasso regularization remains a 
highly competitive and more deployable alternative.

Beyond prediction, the analysis highlights that **opinion 
features, doctor recommendations and healthcare access** 
are the strongest drivers of vaccination behavior all 
actionable targets for public health intervention.

---

## Author

**Naomi Opiyo**
MORINGA School Data Science Program
Phase 3 Project — 2026
