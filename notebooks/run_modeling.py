
import numpy as np
import csv

PROCESSED_DATA_PATH = '../data/processed/airbnb_processed.csv'

def load_processed_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return header, np.array(data, dtype=float)

print("Loading processed data...")
header, data = load_processed_data(PROCESSED_DATA_PATH)
feature_names = header[:-1]
target_name = header[-1]

X_all = data[:, :-1]
Y_all = data[:, -1]

print("Features:", feature_names)
print("Target:", target_name)
print("Data Shape:", X_all.shape)

# Setup and Splitting
# Shuffle the data using a fixed seed for reproducibility
np.random.seed(42)
indices = np.arange(len(X_all))
np.random.shuffle(indices)

test_size = 0.2
split_idx = int(len(X_all) * (1 - test_size))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train_full = X_all[train_indices]
Y_train = Y_all[train_indices]

X_test_full = X_all[test_indices]
Y_test = Y_all[test_indices]

print(f"Training Set: {X_train_full.shape[0]} samples")
print(f"Test Set: {X_test_full.shape[0]} samples")

# Model Implementation
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def train_linear_regression(X, Y):
    # X should already have intercept if needed
    X_T = X.T
    try:
        # (X^T * X)^-1 * X^T * Y
        W = np.linalg.inv(X_T @ X) @ X_T @ Y
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        W = np.linalg.pinv(X_T @ X) @ X_T @ Y
    return W

def predict(X, W):
    return X @ W

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# K-Fold Cross Validation
def k_fold_cross_validation(X, Y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    
    mse_scores = []
    r2_scores = []
    
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train_fold, Y_train_fold = X[train_idx], Y[train_idx]
        X_val_fold, Y_val_fold = X[val_idx], Y[val_idx]
        
        # Train
        W = train_linear_regression(X_train_fold, Y_train_fold)
        
        # Evaluate
        Y_pred = predict(X_val_fold, W)
        mse_val = mse(Y_val_fold, Y_pred)
        r2_val = r2_score(Y_val_fold, Y_pred)
        
        mse_scores.append(mse_val)
        r2_scores.append(r2_val)
        
    return np.mean(mse_scores), np.mean(r2_scores)

# Model 1: Reduced Features
reduced_features = [
    'min_nights_std', 
    'availability_365_std', 
    'review_density_std'
]
# Find OHE columns for Room Types
rt_features = [f for f in feature_names if 'Private room' in f or 'Entire home' in f]
reduced_features += rt_features

print("Selected Reduced Features:", reduced_features)

reduced_indices = [feature_names.index(f) for f in reduced_features]

# Prepare Reduced X (Must add intercept to each fold inside CV? No, the function assumes X passed has what we need?
# Wait, train_linear_regression assumes X has intercept?
# My implementation of train_linear_regression does NOT add intercept. It assumes X has it.
# So I must add intercept to the X passed to K-Fold.

X_train_red = add_intercept(X_train_full[:, reduced_indices])
X_test_red = add_intercept(X_test_full[:, reduced_indices])

# Validate with CV
print("Running CV for Model 1...")
mse_cv_red, r2_cv_red = k_fold_cross_validation(X_train_red, Y_train, k=5)
print(f"Model 1 (Reduced) - 5-Fold CV Results: Avg MSE = {mse_cv_red:.4f}, Avg R2 = {r2_cv_red:.4f}")

# Model 2: Full Features
X_train_all_feat = add_intercept(X_train_full)
X_test_all_feat = add_intercept(X_test_full)

# Validate with CV
print("Running CV for Model 2...")
mse_cv_full, r2_cv_full = k_fold_cross_validation(X_train_all_feat, Y_train, k=5)
print(f"Model 2 (Full) - 5-Fold CV Results: Avg MSE = {mse_cv_full:.4f}, Avg R2 = {r2_cv_full:.4f}")

# Final Evaluation
print("---- Final Test Set Results ----")

# Model 1 Final
W_red = train_linear_regression(X_train_red, Y_train)
Y_pred_red = predict(X_test_red, W_red)
final_mse_red = mse(Y_test, Y_pred_red)
final_r2_red = r2_score(Y_test, Y_pred_red)

# Model 2 Final
W_full = train_linear_regression(X_train_all_feat, Y_train)
Y_pred_full = predict(X_test_all_feat, W_full)
final_mse_full = mse(Y_test, Y_pred_full)
final_r2_full = r2_score(Y_test, Y_pred_full)

print(f"Model 1 (Reduced): MSE = {final_mse_red:.4f}, R2 = {final_r2_red:.4f}")
print(f"Model 2 (Full)   : MSE = {final_mse_full:.4f}, R2 = {final_r2_full:.4f}")

improvement = final_r2_full - final_r2_red
print(f"\nImprovement using Full Features: +{improvement:.4f} R2 score")
