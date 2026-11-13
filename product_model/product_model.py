
import joblib
import pandas as pd

model = joblib.load("models/product_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
median_rating = joblib.load("models/median_rating.pkl")

USE_SCALER = False         
if USE_SCALER:
    scaler = joblib.load("models/scaler.pkl")


def prepare_row(customer):
    customer = dict(customer)

    rating = customer.get("customer_rating")
    if rating is None or pd.isna(rating):
        customer["customer_rating"] = median_rating

    row = pd.DataFrame([customer])
    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    if USE_SCALER:
        row = scaler.transform(row)

    return row


def predict_product(customer):
    return model.predict(prepare_row(customer))[0]


def predict_with_confidence(customer, min_confidence=0.35):
    probs = model.predict_proba(prepare_row(customer))[0]
    order = probs.argsort()[::-1]
    return {
        "product": model.classes_[order[0]],
        "confidence": round(float(probs[order[0]]), 3),
        "runner_up": model.classes_[order[1]],
        "runner_up_confidence": round(float(probs[order[1]]), 3),
        "low_confidence": bool(probs[order[0]] < min_confidence),
    }

if __name__ == "__main__":
    test_example = {
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
    print("Predicted product:", predict_product(test_example))
    print(predict_with_confidence(test_example))
