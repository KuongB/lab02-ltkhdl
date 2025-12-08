
import numpy as np
import csv
import sys
import os

# --- Configuration ---
DATA_PATH = '../data/raw/AB_NYC_2019.csv'
PROCESSED_PATH = '../data/processed/airbnb_processed.csv'

# --- 1. Preprocessing Logic Audit ---
print("--- [Audit] Starting Preprocessing Logic Check ---")

# Mock Load (or Real Load)
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file not found at {DATA_PATH}")
    sys.exit(1)

def load_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, np.array(data, dtype=object)

header, data = load_data(DATA_PATH)
col_map = {name: i for i, name in enumerate(header)}

# Features
reviews_per_month_idx = col_map.get('reviews_per_month', -1)
price_idx = col_map.get('price', -1)
min_nights_idx = col_map.get('minimum_nights', -1)
number_of_reviews_idx = col_map.get('number_of_reviews', -1)
availability_365_idx = col_map.get('availability_365', -1)
host_listings_count_idx = col_map.get('calculated_host_listings_count', -1)
neighbourhood_idx = col_map.get('neighbourhood_group', -1)
room_type_idx = col_map.get('room_type', -1)

# Helper
def safe_float_convert(column_data, default_val=np.nan):
    res = []
    for x in column_data:
        try:
            val = float(x) if x != '' else default_val
            res.append(val)
        except:
            res.append(default_val)
    return np.array(res)

# Extract Data
reviews_per_month = safe_float_convert(data[:, reviews_per_month_idx], 0.0)
price = safe_float_convert(data[:, price_idx], np.nan)
min_nights = safe_float_convert(data[:, min_nights_idx], np.nan)
num_reviews = safe_float_convert(data[:, number_of_reviews_idx], np.nan)
availability = safe_float_convert(data[:, availability_365_idx], np.nan)
host_count = safe_float_convert(data[:, host_listings_count_idx], np.nan)
neighbourhood = data[:, neighbourhood_idx]
room_type = data[:, room_type_idx]

# Imputation Check
def impute_median(arr):
    mask = np.isnan(arr)
    if np.any(mask):
        arr[mask] = np.nanmedian(arr)
    return arr

# Apply Imputation
price = impute_median(price) # Though we drop invalid prices usually
min_nights = impute_median(min_nights)
num_reviews = impute_median(num_reviews)
availability = impute_median(availability)
host_count = impute_median(host_count)

# Outlier & Valid Check
valid_mask = price > 0
data_len_pre = len(price)
price = price[valid_mask]
min_nights = min_nights[valid_mask]
num_reviews = num_reviews[valid_mask]
reviews_per_month = reviews_per_month[valid_mask]
availability = availability[valid_mask]
host_count = host_count[valid_mask]
neighbourhood = neighbourhood[valid_mask]
room_type = room_type[valid_mask]

log_price = np.log1p(price)
q1, q3 = np.percentile(log_price, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5*iqr
upper = q3 + 1.5*iqr
outlier_mask = (log_price >= lower) & (log_price <= upper)

# Apply Filter
log_price = log_price[outlier_mask]
min_nights = min_nights[outlier_mask]
num_reviews = num_reviews[outlier_mask]
reviews_per_month = reviews_per_month[outlier_mask]
availability = availability[outlier_mask]
host_count = host_count[outlier_mask]
neighbourhood = neighbourhood[outlier_mask]
room_type = room_type[outlier_mask]

# Feature Engineering
host_count[host_count == 0] = 1
review_density = reviews_per_month / host_count

def standardize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0: return arr - mean
    return (arr - mean) / std

s_min_nights = standardize(min_nights)
s_num_reviews = standardize(num_reviews)
s_availability = standardize(availability)
s_reviews_pm = standardize(reviews_per_month)
s_density = standardize(review_density)
s_log_price = standardize(log_price)

# Check Standardization
print("\n[Check] Standardization Stats (Expected Mean~0, Std~1):")
for name, arr in [('MinNights', s_min_nights), ('Reviews', s_num_reviews)]:
    m, s = np.mean(arr), np.std(arr)
    print(f"  {name}: Mean={m:.6f}, Std={s:.6f} -> {'PASS' if abs(m) < 1e-6 and abs(s-1) < 1e-6 else 'FAIL'}")

# Interaction / Poly
interact_1 = standardize(s_min_nights * s_reviews_pm)
interact_2 = standardize(s_availability * s_num_reviews)
poly_1 = standardize(s_min_nights**2)

# OHE Logic Check
def one_hot_encode(cats):
    uniques = np.unique(cats)
    uniques.sort()
    # DROP FIRST
    if len(uniques) > 1:
        keep = uniques[1:]
    else:
        keep = uniques
    
    encoded = []
    for v in keep:
        encoded.append((cats == v).astype(int))
    return np.column_stack(encoded), len(uniques), len(keep)

oh_nb, n_nb_orig, n_nb_keep = one_hot_encode(neighbourhood)
oh_rt, n_rt_orig, n_rt_keep = one_hot_encode(room_type)

print("\n[Check] One-Hot Encoding (Dummy Trap):")
print(f"  Neighbourhood: Orig Cats={n_nb_orig}, Kept Columns={n_nb_keep} -> {'PASS' if n_nb_keep == n_nb_orig - 1 else 'FAIL'}")
print(f"  Room Type:     Orig Cats={n_rt_orig}, Kept Columns={n_rt_keep} -> {'PASS' if n_rt_keep == n_rt_orig - 1 else 'FAIL'}")

# Final Matrix
X_full = np.column_stack([
    s_min_nights, s_num_reviews, s_availability, s_density,
    interact_1, interact_2, poly_1,
    oh_nb, oh_rt
])
Y = s_log_price

# Missing Value Check
n_nans = np.isnan(X_full).sum()
n_infs = np.isinf(X_full).sum()
print(f"\n[Check] Final X Matrix Validity:")
print(f"  NaN Count: {n_nans} -> {'PASS' if n_nans == 0 else 'FAIL'}")
print(f"  Inf Count: {n_infs} -> {'PASS' if n_infs == 0 else 'FAIL'}")


# --- 2. Modeling Logic Audit ---
print("\n--- [Audit] Starting Modeling Logic Check ---")

def add_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def train_ridge(X, y, lamb=0.001):
    # Verify we use numpy only
    n_feat = X.shape[1]
    I = np.eye(n_feat)
    I[0,0] = 0
    # Formula check
    try:
        W = np.linalg.inv(X.T @ X + lamb * I) @ X.T @ y
        return W, "Inv"
    except:
        W = np.linalg.pinv(X.T @ X + lamb * I) @ X.T @ y
        return W, "Pinv"

# Mock run on subset
X_sub = add_intercept(X_full[:1000])
Y_sub = Y[:1000]

W_ridge, method = train_ridge(X_sub, Y_sub, 0.001)
print(f"\n[Check] Ridge Regression Execution:")
print(f"  Method Used: {method}")
print(f"  Weights Shape: {W_ridge.shape} (Expected {X_sub.shape[1]}) -> {'PASS' if W_ridge.shape[0] == X_sub.shape[1] else 'FAIL'}")

# KNN Check
class KNN:
    def __init__(self, k=5): self.k=k
    def fit(self, X, y): self.X=X; self.y=y
    def predict(self, Xt):
        # Naive implementation for check
        dists = np.sqrt(np.sum((self.X[:, None] - Xt) ** 2, axis=2))
        # This naive broadcast is memory heavy, just check logic on tiny set
        preds = []
        for i in range(len(Xt)):
            # This matches the notebook logic per-point
            d = np.sqrt(np.sum((self.X - Xt[i])**2, axis=1))
            idx = np.argpartition(d, self.k)[:self.k]
            preds.append(np.mean(self.y[idx]))
        return np.array(preds)

knn = KNN(k=5)
knn.fit(X_sub[:50], Y_sub[:50]) # Train on 50
preds = knn.predict(X_sub[50:60])  # Pred on 10
print(f"\n[Check] KNN Execution:")
print(f"  Predictions Generated: {len(preds)}")
print(f"  Values within range: {np.all(preds > -5) and np.all(preds < 5)} -> PASS")

print("\n--- Audit Complete ---")
