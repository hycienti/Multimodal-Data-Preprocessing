"""Build the product model's input for a verified member."""
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEMBERS_FILE = Path(__file__).resolve().parent / "members.json"
DATASET_FILE = ROOT / "product_model" / "data" / "merged_dataset.csv"

PROFILE_FIELDS = [
    "purchase_amount", "customer_rating", "engagement_score", "purchase_interest_score",
    "num_profiles", "main_platform", "main_sentiment",
]


def customer_id_for(member):
    members = json.loads(MEMBERS_FILE.read_text())
    customer_id = members.get(member.strip().lower())
    if customer_id is None:
        raise KeyError(f"No customer_id mapped for '{member}' in {MEMBERS_FILE.name}")
    return int(customer_id)


def customer_profile(customer_id, today=None):
    """Return the feature dict for a customer: their latest transaction and social profile,
    dated today (month / day_of_week of the prediction, not of the old purchase)."""
    data = pd.read_csv(DATASET_FILE, parse_dates=["purchase_date"])
    rows = data[data["customer_id"] == customer_id]
    if rows.empty:
        raise KeyError(f"customer_id {customer_id} not found in {DATASET_FILE.name}")

    latest = rows.sort_values("purchase_date").iloc[-1]
    profile = {field: latest[field] for field in PROFILE_FIELDS}
    profile["customer_rating"] = None if pd.isna(profile["customer_rating"]) else float(profile["customer_rating"])

    today = today or date.today()
    profile["month"] = today.month
    profile["day_of_week"] = today.weekday()
    return profile
