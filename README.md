# Customer Churn Prediction System

An end-to-end customer-retention analysis using Python, scikit-learn,
Streamlit, and Power BI. The project explores telecom customer behaviour,
trains classification models, and provides a simple interface for evaluating
churn risk.

## Business Problem

Customer churn reduces recurring revenue and increases acquisition costs. This
project identifies patterns associated with churn so a retention team can
prioritise customers who may need intervention.

## Project Workflow

1. Clean and validate the Telco Customer Churn dataset.
2. Explore customer tenure, contracts, charges, and churn behaviour.
3. Prepare model-ready features.
4. Compare classification models.
5. Save the selected model for inference.
6. Present all nineteen model inputs in a Streamlit interface.

## Results

- Approximately 80% test accuracy was recorded with the reproducible training
  script.
- The analysis investigates tenure, monthly charges, and contract type as
  customer-risk indicators.
- The Streamlit application converts model output into a clear churn/stay
  result.

Accuracy alone does not fully describe model quality. Precision, recall, and
class balance should also be reviewed before using a churn model operationally.

## Repository Structure

| Path | Purpose |
|---|---|
| `Notebooks/churn_project.ipynb` | Cleaning, exploration, training, and evaluation |
| `App/streamlit_app.py` | Interactive churn prediction interface |
| `Models/churn_model.pkl` | Serialized trained model |
| `Data/` | Raw and cleaned analysis datasets |
| `train_model.py` | Reproducible model-training entry point |

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run App/streamlit_app.py
```

## Skills Demonstrated

`Python` `Pandas` `scikit-learn` `Streamlit` `EDA` `Classification`
`Customer Analytics`

## Responsible Use

This is a portfolio project built for learning and demonstration. A production
retention decision should include model monitoring, fairness checks, current
customer data, and human review.

## Author

Adarsh Dubey - Data Analyst
