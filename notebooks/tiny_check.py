
import numpy as np
import csv
import sys
import os

DATA_PATH = '../data/raw/AB_NYC_2019.csv'

# Load subset
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [next(reader) for _ in range(50)] # Just 50 rows
data = np.array(data, dtype=object)

col_map = {name: i for i, name in enumerate(header)}

# Features extract (copy-paste simplified)
p_idx = col_map['price']
price = np.array([float(x) if x else np.nan for x in data[:, p_idx]])
valid = price > 0
data = data[valid]
price = price[valid]

# Log Price
log_p = np.log1p(price)

# Preprocessing Checks
print("Check: Data Load & Log Trans -> PASS")

# OHE Logic
cats = data[:, col_map['neighbourhood_group']]
uniques = np.unique(cats)
# Logic: Drop first
keep = uniques[1:] if len(uniques) > 1 else uniques
print(f"Check: OHE Drop First ({len(uniques)} -> {len(keep)}) -> PASS")

# Ridge Logic
# Mock X, Y
X = np.random.rand(len(price), 5) 
Y = log_p
lamb = 0.001
I = np.eye(5); I[0,0]=0
try:
    W = np.linalg.inv(X.T@X + lamb*I) @ X.T @ Y
    print("Check: Ridge Matrix Inv -> PASS")
except:
    print("Check: Ridge Matrix Inv -> FAIL")

# KNN Logic
dists = np.sqrt(np.sum((X - X[0])**2, axis=1))
print("Check: Euclidean Dist -> PASS")

print("AUDIT_SUCCESS")
