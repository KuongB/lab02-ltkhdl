
import numpy as np
import csv
import os

DATA_PATH = '../data/raw/AB_NYC_2019.csv'

def load_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, np.array(data, dtype=object)

print("Starting data load...")
header, data = load_data(DATA_PATH)
print(f"Data Loaded. Shape: {data.shape}")
col_map = {name: i for i, name in enumerate(header)}

# Ensure output directory exists
os.makedirs('../data/processed', exist_ok=True)

# Identify columns
reviews_per_month_idx = col_map['reviews_per_month']
price_idx = col_map['price']
min_nights_idx = col_map['minimum_nights']
number_of_reviews_idx = col_map['number_of_reviews']
availability_365_idx = col_map['availability_365']
host_listings_count_idx = col_map['calculated_host_listings_count']

# Helper to safely convert to float and handle empty strings (which act as missing in CSV load)
def safe_float_convert(column_data, default_val=np.nan):
    res = []
    for x in column_data:
        try:
            if x == '':
                res.append(default_val)
            else:
                res.append(float(x))
        except:
            res.append(default_val)
    return np.array(res)

# 1. Impute reviews_per_month -> 0
reviews_per_month_data = safe_float_convert(data[:, reviews_per_month_idx], default_val=0.0)

# 2. Other numerics -> Median
price_data = safe_float_convert(data[:, price_idx], default_val=np.nan)
min_nights_data = safe_float_convert(data[:, min_nights_idx], default_val=np.nan)
number_of_reviews_data = safe_float_convert(data[:, number_of_reviews_idx], default_val=np.nan)
availability_365_data = safe_float_convert(data[:, availability_365_idx], default_val=np.nan)
host_listings_count_data = safe_float_convert(data[:, host_listings_count_idx], default_val=np.nan)

def impute_median(arr):
    mask = np.isnan(arr)
    if np.any(mask):
        median_val = np.nanmedian(arr)
        print(f"Imputing {np.sum(mask)} missing values with median: {median_val}")
        arr[mask] = median_val
    return arr

price_data = impute_median(price_data)
min_nights_data = impute_median(min_nights_data)
number_of_reviews_data = impute_median(number_of_reviews_data)
availability_365_data = impute_median(availability_365_data)
host_listings_count_data = impute_median(host_listings_count_data)

# Log transform Price
valid_price_mask = price_data > 0
price_data = price_data[valid_price_mask]

# Sync other arrays to this mask
min_nights_data = min_nights_data[valid_price_mask]
number_of_reviews_data = number_of_reviews_data[valid_price_mask]
reviews_per_month_data = reviews_per_month_data[valid_price_mask]
availability_365_data = availability_365_data[valid_price_mask]
host_listings_count_data = host_listings_count_data[valid_price_mask]
# Also sync categorical columns for later
neighbourhood_data = data[valid_price_mask, col_map['neighbourhood_group']]
room_type_data = data[valid_price_mask, col_map['room_type']]

# Apply Log1p
log_price = np.log1p(price_data)

# Remove outliers using IQR on log_price (standard method)
q1 = np.percentile(log_price, 25)
q3 = np.percentile(log_price, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (log_price >= lower_bound) & (log_price <= upper_bound)
print(f"Retaining {np.sum(outlier_mask)} / {len(log_price)} samples after outlier removal.")

# Apply mask
log_price = log_price[outlier_mask]
min_nights_data = min_nights_data[outlier_mask]
number_of_reviews_data = number_of_reviews_data[outlier_mask]
reviews_per_month_data = reviews_per_month_data[outlier_mask]
availability_365_data = availability_365_data[outlier_mask]
host_listings_count_data = host_listings_count_data[outlier_mask]
neighbourhood_data = neighbourhood_data[outlier_mask]
room_type_data = room_type_data[outlier_mask]

# Feature Engineering
safe_denominator = host_listings_count_data.copy()
safe_denominator[safe_denominator == 0] = 1 
review_density = reviews_per_month_data / safe_denominator

# Standardization (Z-Score)
def standardize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return arr - mean # Centered but not scaled
    return (arr - mean) / std

scaled_min_nights = standardize(min_nights_data)
scaled_num_reviews = standardize(number_of_reviews_data)
scaled_availability = standardize(availability_365_data)
scaled_review_density = standardize(review_density)
scaled_log_price = standardize(log_price)

# One-Hot Encoding
def one_hot_encode(categoricals):
    unique_vals = np.unique(categoricals)
    unique_vals.sort()
    encoding = []
    for val in unique_vals:
        col = (categoricals == val).astype(int)
        encoding.append(col)
    return np.column_stack(encoding), unique_vals

oh_neighbourhood, nb_names = one_hot_encode(neighbourhood_data)
oh_room_type, rt_names = one_hot_encode(room_type_data)

print("Neighbourhood Features:", nb_names)

# Create Full Feature Matrix X
# Order: scaled_min_nights, scaled_num_reviews, scaled_availability, scaled_review_density, [Neighborhood OHE], [RoomType OHE]
X_full = np.column_stack([
    scaled_min_nights,
    scaled_num_reviews,
    scaled_availability,
    scaled_review_density,
    oh_neighbourhood,
    oh_room_type
])

Y = scaled_log_price

feature_names = ['min_nights_std', 'num_reviews_std', 'availability_365_std', 'review_density_std'] + list(nb_names) + list(rt_names)

print("X_full shape:", X_full.shape)
print("Y shape:", Y.shape)

output_file = '../data/processed/airbnb_processed.csv'

# Combine X and Y (Y as last column usually)
final_data = np.column_stack([X_full, Y])
final_header = feature_names + ['target_log_price_std']

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(final_header)
    writer.writerows(final_data)

print(f"Saved processed data to {output_file}")
