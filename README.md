# Uplift Modeling & Marketing Campaign Optimization

An end-to-end causal machine learning project built to optimize marketing campaigns by predicting individual customer response and maximizing return on investment (ROI).

---

## Project Overview
Traditional machine learning models predict overall purchase probability, which often leads to wasted marketing budgets on:
* **Sure Things:** Customers who would have purchased anyway.
* **Sleeping Dogs:** Customers who get annoyed by ads and are more likely to churn.

This project implements Causal Machine Learning using an X-Learner meta-learning architecture to isolate Individual Treatment Effects (CATE), allowing businesses to target strictly Persuadables.

---

## Key Features
* **Synthetic Data Generation:** Simulated a randomized control trial (RCT) customer dataset with features, treatment assignments, and conversion outcomes.
* **Causal Meta-Learner:** Built an EconML XLearner pipeline utilizing LightGBM base estimators to compute individual uplift scores.
* **Model Evaluation:** Generated cumulative gain Qini Curves to validate that the model outperforms random baseline targeting.
* **Budget Simulation:** Calculated incremental conversions under custom marketing budget constraints.

---

## Tech Stack
* Python
* Pandas & NumPy
* LightGBM
* EconML
* Matplotlib

---

## Results & Evaluation
The model's performance was evaluated using a cumulative gain Qini Curve. The resulting chart proves that targeting users based on high uplift scores captures significantly more incremental conversions early on compared to random baseline guessing.

---

## Repository Structure
```text
├── uplift_modeling.ipynb    # Main Google Colab notebook with code & analysis
└── README.md                # Project documentation
