# Product Recommendation Model

Predicts which product category (Sports, Electronics, Clothing, Groceries or Books) a customer will buy, using their social media profile and their transaction data.

## Folder structure

```
product_model/
├── README.md
├── requirements.txt
├── .gitignore
├── product_model.py  
├── product_recommendation_model.ipynb
├── data/
│   ├── customer_social_profiles - customer_social_profiles.csv
│   ├── customer_transactions - customer_transactions.csv
│   ├── merged_dataset.csv
│   └── model_ready_dataset.csv
└── models/
    ├── product_model.pkl
    ├── feature_columns.pkl
    ├── median_rating.pkl
    └── scaler.pkl

| File | Description |
|---|---|
| `notebooks/product_recommendation_model.ipynb` | Full notebook: cleaning, merge, EDA, features, models, evaluation |
| `product_model.py` | Prediction script, loads the saved files from `models/` |
| `data/merged_dataset.csv` | Merged data with engineered features, 117 rows, 8 missing ratings kept on purpose |
| `data/model_ready_dataset.csv` | Encoded features and target, ratings filled, no missing values |
| `data/customer_*.csv` | Original social profile and transaction data |
| `models/product_model.pkl` | Trained Random Forest |
| `models/feature_columns.pkl` | Column names and order the model expects |
| `models/median_rating.pkl` | Training median used to fill a missing rating |
| `models/scaler.pkl` | Scaler fitted on the training data (only used by Logistic Regression) |

## Data preparation

- Removed 5 duplicate rows from the social profiles (155 to 150 rows).
- The join key was written differently in each file (`A178` and `178`), so I removed the letter and converted it to a whole number.
- The social table had several rows per customer (84 unique customers). I summarised it to one row per customer: the average of the two scores, the most common platform and sentiment, and a count of profile rows (`num_profiles`).
- Inner join on `customer_id`: 150 transactions became 117 rows. The other 33 belong to customers with no social profile.
- 8 ratings are missing. They are filled with the training-set median (3.5) after the train/test split, to avoid leakage.
- Features: `purchase_amount`, `customer_rating`, `engagement_score`, `purchase_interest_score`, `num_profiles`, `month`, `day_of_week`, plus one-hot encoded `main_platform` and `main_sentiment` (15 columns in total).

## Results

Single 80/20 split (24 test rows):

| Model | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|
| Baseline  | 0.250 | 0.100 | 1.593 |
| Logistic Regression | 0.167 | 0.160 | 2.428 |
| Random Forest | 0.167 | 0.161 | 1.669 |

5-fold cross-validation (more reliable with this little data):

| Model | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|
| Baseline  | 0.214 | 0.075 | 1.600 |
| Logistic Regression | 0.240 | 0.223 | 1.939 |
| Random Forest | 0.290 | 0.275 | 1.593 |

Random Forest was chosen because it is best on all three metrics in cross-validation. Its lead over the baseline is small, so the features carry weak signal.

## Limitations

- Only 117 rows, about 23 per product, so results vary a lot between splits.
- `purchase_amount` and `customer_rating` describe the purchase itself, so a real system would not know them before the customer chooses a product.

## How to use

Install the requirements, then import the script. It finds the `models/` folder on its own, so it works from any directory.

```
pip install -r requirements.txt
```

```python
from product_model import predict_product, predict_with_confidence

customer = {
    "purchase_amount": 250,
    "customer_rating": 3.5,        
    "engagement_score": 80,
    "purchase_interest_score": 4.0,
    "num_profiles": 1,
    "month": 3,
    "day_of_week": 2,              
    "main_platform": "Twitter",
    "main_sentiment": "Positive",
}

print(predict_product(customer))
print(predict_with_confidence(customer))
```

`predict_with_confidence` returns the top product, its probability, the runner-up, and a `low_confidence` flag (set when the probability is below 0.35).
