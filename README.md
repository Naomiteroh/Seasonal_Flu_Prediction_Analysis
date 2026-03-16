# Seasonal Flu Vaccine Prediction
## A Machine Learning Classification Project

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Project Overview

This project applies the CRISP-DM (Cross-Industry Standard 
Process for Data Mining) framework to predict whether an 
individual will receive the seasonal flu vaccine based on 
their background, opinions and health behaviors.

While the data originates from the National 2009 H1N1 Flu 
Survey conducted in the United States, the methodology and 
findings are adapted for the **Kenyan public health context**, 
where seasonal flu vaccination remains a recurring challenge 
across two annual flu seasons.

The final deliverables include:
- Three comprehensive Jupyter notebooks following the CRISP-DM framework
- A complete machine learning classification pipeline
- Actionable stakeholder recommendations for Kenyan public health



##  Business Problem

Kenya experiences two flu seasons annually that is March to May 
and October to December. Seasonal flu vaccination rates 
remain low particularly among younger and lower income 
populations.

This project answers:
- Who is least likely to get vaccinated and why?
- Which behavioral, opinion and demographic factors 
  predict vaccination?
- How can public health departments and hospitals use 
  these insights to design targeted interventions?

**Key Stakeholders:**
| Stakeholder | Need |

| County Director of Health (CDoH) | Identify which communities to target for outreach |
| CEO, County Referral Hospital | Identify which patients need clinical encouragement |



##  Data

The dataset comes from the **National 2009 H1N1 Flu Survey** 
collected by the United States government.

Link: [DrivenData — Flu Shot Learning](https://www.drivendata.org/competitions/66/flu-shot-learning/)

| File | Description | Used |

| `training_set_features.csv` | 35 features for 26,707 respondents |  Yes |

| `training_set_labels.csv` | Target variables |  Yes - seasonal_vaccine only |

| `test_set_features.csv` | Unseen respondent features |  No - competition only |

| `submission_format.csv` | Competition submission template |  No - not competing |

**Target Variable:** `seasonal_vaccine` (binary: 0 = not vaccinated, 1 = vaccinated)

**Class Balance:** 53% not vaccinated / 47% vaccinated - balanced dataset



##  Repository Structure
```
Phase_3_Project/
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
├── Presentation/
│   └── Seasonal_flu_prediction_analysis_presentation.pdf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

##  Notebooks

Run the notebooks in the following order:

| Order | Notebook | Purpose | Output |

| 1 | [01_Data_cleaning.ipynb](Notebooks/01_Data_cleaning.ipynb) | Business understanding, data understanding, duplicate removal, missing values visualization | `cleaned_data.csv` |

| 2 | [02_Analysis.ipynb](Notebooks/02_Analysis.ipynb) | EDA, feature distributions, outlier detection, feature vs target analysis, correlation analysis, feature selection | Visualizations and feature selection decisions |

| 3 | [03_Final_notebook.ipynb](Notebooks/03_Final_notebook.ipynb) | Complete ML pipeline, modeling, evaluation, recommendations | `processed_data.csv`, trained models, recommendations |



##  CRISP-DM Framework Application

### 1️⃣ Business Understanding
Defined the public health problem, identified two Kenyan 
public health stakeholders and framed the project as a 
binary classification problem predicting seasonal flu 
vaccination behavior.

### 2️⃣ Data Understanding
Explored the National 2009 H1N1 Flu Survey dataset:
- 26,707 respondents and 35 features
- Balanced class distribution (53/47)
- 30 out of 36 columns with missing values identified
- No duplicates found

### 3️⃣ Data Preparation
- Dropped 14 irrelevant features including all H1N1 specific 
  columns, negligible predictors and columns not transferable 
  to the Kenyan context
- Train test split (80/20) with stratification
- Median imputation for numeric features
- Most frequent imputation for categorical features
- StandardScaler for numeric features
- OneHotEncoder for categorical features
- All transformers fitted on training data only to prevent 
  data leakage

### 4️⃣ Modeling
Five models built iteratively:
- Logistic Regression (baseline)
- Logistic Regression + Ridge (L2 regularization)
- Logistic Regression + Lasso (L1 regularization)
- Decision Tree (nonparametric)
- Random Forest (ensemble) ← best model

### 5️⃣ Evaluation
Primary metric: ROC AUC Score

| Model | ROC AUC |

| Logistic Regression Baseline | 0.8506 |

| Logistic Regression + Ridge | 0.8506 |

| Logistic Regression + Lasso | 0.8506 |

| Decision Tree | 0.8356 |

| **Random Forest** | **0.8518** |

#### Beyond Metrics Evaluation
| Factor | Random Forest | Logistic Regression |

| ROC AUC | 0.8518  | 0.8506 |

| Runtime | Slow  | Fast  |

| Explainability | Low  | High  |

| Parsimony | Low  | High  |

| Ease of Deployment | Complex  | Simple  |

The ROC AUC difference is only 0.0012 — practically negligible.
In a resource constrained Kenyan clinical setting, Logistic 
Regression with Lasso is a strong alternative given its 
explainability and ease of deployment.

##  Key Findings

- **Opinion features are the strongest predictors** - perceived 
  flu risk (r=0.390), doctor recommendation (r=0.369) and 
  vaccine effectiveness belief (r=0.362) are the top three 
  predictors
- **Age is the strongest demographic predictor** - 65+ years: 
  67.4% vaccinated vs 18-34 years: only 28.5%
- **Uncertainty blocks vaccination** - people who say 
  "don't know" about vaccine side effects have the lowest 
  vaccination rate at 18.1%
- **Healthcare access drives uptake** - health insurance 
  and chronic conditions are strong predictors
- **Income predicts vaccination** - below poverty: 36.3% vs 
  above poverty: 49.7%

---

##  Strategic Recommendations

**For County Director of Health:**
- Target low risk perception communities with flu risk 
  education campaigns
- Design youth targeted vaccination programs - 18-34 year 
  olds are significantly underserved
- Remove financial barriers by providing free vaccination 
  at public health facilities
- Counter vaccine misinformation using trusted community leaders

**For County Referral Hospital CEO:**
- Mandate doctor recommendations for seasonal flu vaccination 
  during every clinical consultation
- Flag and proactively contact uninsured patients before 
  each flu season
- Integrate seasonal flu vaccination into chronic disease 
  management protocols
- Train clinicians to address side effect uncertainty 
  not just worry


## **Non Technical Presentation**
The non-technical presentation can be found in the Presentation folder.
Link: https://github.com/Naomiteroh/Seasonal_Flu_Prediction_Analysis/blob/main/Presentation/Seasonal_flu_prediction_analysis_presentation.pdf


##  Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Naomiteroh/Seasonal_Flu_Prediction_Analysis.git
```

**2. Create a virtual environment (recommended)**
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

**5. Run Jupyter Notebook**
```bash
jupyter notebook
```

Open notebooks in order: `01_Data_cleaning` → `02_Analysis` → `03_Final_notebook`



**NAOMI OPIYO**
MORINGA School Data Science Program  
Phase 3 Project — 2026
