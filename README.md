# Uplift Modeling & Marketing Campaign Optimization

An end-to-end causal machine learning project built to optimize marketing campaigns by predicting individual customer response and maximizing return on investment (ROI) using real-world e-commerce data.

---

## Project Overview
Traditional machine learning models predict overall purchase probability, which often leads to wasted marketing budgets on:
* **Sure Things:** Customers who would have purchased anyway.
* **Sleeping Dogs:** Customers who get annoyed by ads and are more likely to churn.

This project implements Causal Machine Learning using an X-Learner meta-learning architecture to isolate Individual Treatment Effects (CATE), allowing businesses to target strictly Persuadables.

---

## Dataset
* **Source:** Kevin Hillstrom MineThatData E-Mail Analytics Dataset (Kaggle).
* **Scale:** 64,000 customer records from a randomized controlled trial (RCT).
* **Treatment ($T$):** Customers split across promotional email campaigns vs. a control group (`No E-Mail`).
* **Outcome ($Y$):** Customer conversion and purchase behavior.

---

## Key Features & Steps
* **Data Processing:** Cleaned 64k records and mapped email segments into binary treatment and control groups.
* **Feature Engineering:** One-hot encoded categorical customer history variables (recency, spending history segments, zip codes, and shopping channels).
* **Causal Meta-Learner:** Built an EconML XLearner pipeline utilizing LightGBM base estimators to compute individual uplift scores (CATE).
* **Model Evaluation:** Generated cumulative gain Qini Curves and calculated the Area Under the Uplift Curve (AUUC) score.
* **Financial ROI Simulation:** Simulated corporate budget constraints (e.g., 20% target cap) to prove net profit gains over random baseline targeting.

---

## Tech Stack
* Python
* Pandas & NumPy
* LightGBM
* EconML
* Matplotlib

---

## Repository Structure
```text
├── uplift_modeling.ipynb    # Main Google Colab notebook with code & analysis
└── README.md                # Project documentation
