# NYC Airbnb Price Prediction: A NumPy-Only Approach

## 1. Project Overview
This project focuses on analyzing and predicting Airbnb listing prices in New York City using the **2019 NYC Airbnb Open Data**. 

The core technical challenge and distinguishing feature of this project is the **exclusive use of NumPy** for the entire data pipeline—including data loading, cleaning, complex feature engineering, and the implementation of machine learning models (Ordinary Least Squares and Lasso Regression) from scratch. High-level libraries like Pandas or Scikit-learn were intentionally avoided for the core logic to demonstrate a deep understanding of vectorized operations, memory management with structured arrays, and the mathematical foundations of regression algorithms.

## 2. Table of Contents
1. [Project Overview](#1-project-overview)
2. [Introduction](#3-introduction)
3. [Dataset](#4-dataset)
4. [Methodology & Mathematical Foundations](#5-methodology--mathematical-foundations)
5. [NumPy Techniques Used](#6-numpy-techniques-used)
6. [Installation & Setup](#7-installation--setup)
7. [Usage](#8-usage)
8. [Results & Analysis](#9-results--analysis)
9. [Project Structure](#10-project-structure)
10. [Challenges & Solutions](#11-challenges--solutions)
11. [Future Improvements](#12-future-improvements)
12. [Contributors](#13-contributors)
13. [License](#14-license)

## 3. Introduction
### Problem Statement
Airbnb pricing is dynamic and influenced by a multitude of factors including location, room type, availability, and host activity. Predicting an optimal price is challenging for hosts, and understanding fair value is difficult for guests.

### Motivation
By building a predictive model, we can uncover the latent drivers of listing prices. Implementing this "from scratch" allows us to peek under the hood of standard ML libraries, understanding the optimization landscapes and computational bottlenecks involved in processing real-world data.

### Objectives
- Perform Exploratory Data Analysis (EDA) to find pricing trends.
- Preprocess data using strictly **NumPy** (handling missing values, outliers, encoding).
- Implement **Linear Regression (OLS)** and **Lasso Regression (Coordinate Descent)** from mathematical first principles.
- Evaluate models using standard metrics ($MSE$, $R^2$) and compare with Scikit-learn.

## 4. Dataset & Exploratory Analysis
- **Source**: [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- **File**: `AB_NYC_2019.csv`
- **Size**: ~49,000 listings, 16 features.

### 4.1. Key Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| `neighbourhood_group` | Categorical | The 5 boroughs: Manhattan, Brooklyn, Queens, Bronx, Staten Island. |
| `room_type` | Categorical | Entire home, Private room, Shared room. |
| `price` | Continuous | Listing price in USD (Target Variable). |
| `minimum_nights` | Continuous | Minimum number of nights required to book. |
| `number_of_reviews` | Continuous | Total reviews received. |
| `availability_365` | Continuous | Number of days available in the future 365 days. |

*Other features included in the dataset: `id`, `name`, `host_id`, `host_name`, `neighbourhood`, `latitude`, `longitude`, `last_review`, `reviews_per_month`, `calculated_host_listings_count`.*

### 4.2. Key Questions & Insights
I performed extensive EDA (`01_data_exploration.ipynb`) to answer critical business questions:

**Q1: How does location influence price?**
-   **Answer**: Location is the primary driver. **Manhattan** is significantly more expensive (Median: ~$150) compared to **Brooklyn** (~$90) and **Queens** (~$50). The price distribution in Manhattan is also much wider, indicating a mix of luxury and standard options.

**Q2: What is the price premium for privacy?**
-   **Answer**: "Entire home/apt" listings command a huge premium (Median: ~$160) over "Private rooms" (~$70). "Shared rooms" are the cheapest (~$40) but represent a very small niche of the market.

**Q3: Are "popular" listings cheaper or more expensive?**
-   **Answer**: There is a weak **negative correlation** between `price` and `number_of_reviews`. Cheaper listings tend to accumulate more reviews, likely due to higher turnover and affordability. Expensive luxury listings get fewer bookings/reviews.

**Q4: Can we detect "commercial" hosts?**
-   **Answer**: Yes. Hosts with `calculated_host_listings_count` > 1 manage multiple properties. I found that a small percentage of hosts control a large number of listings, suggesting professional property management rather than individual sharing.

**Q5: How is price distributed?**
-   **Answer**: Highly **right-skewed**. Most listings are under $200, but the tail extends to $10,000+. This necessitates the Log-transformation used in my modeling phase.

## 5. Methodology & Mathematical Foundations

### 5.1. Feature Engineering (NumPy)
The preprocessing pipeline (`02_preprocessing.ipynb`) transforms raw CSV data into a clean, numerical matrix $X$:

1.  **Log-Transformation**: The target variable `price` is highly right-skewed. I apply:
    $$ y' = \log(y + 1) $$
    This stabilizes variance and makes the data conform better to the Gaussian assumptions of Linear Regression.

2.  **Interaction Terms**: Capturing non-linear relationships:
    -   `interaction_nights_reviews` = `minimum_nights` $\times$ `reviews_per_month`
    -   `is_high_value_core`: A boolean mask converted to float, isolating high-density areas (Manhattan downtown) based on latitude/longitude boundaries.

3.  **One-Hot Encoding**: Categorical variables (`neighbourhood_group`, `room_type`) are converted into binary vectors using broadcasting comparisons (`data['col'][:, None] == unique_values`). The first category is dropped to prevent perfect collinearity (Dummy Variable Trap).

4.  **Standard Scaling (Z-Score)**:
    For every feature $j$:
    $$ x_{ij}' = \frac{x_{ij} - \mu_j}{\sigma_j} $$
    This ensures uniform gradients during optimization.

### 5.2. Model 1: Ordinary Least Squares (OLS)
**Objective**:
Linear Regression fits a linear model with coefficients $W = (w_0, \dots, w_p)$ to minimize the Residual Sum of Squares (RSS) between the observed targets in the dataset and the targets predicted by the linear approximation.

The objective function (Cost Function) is defined as:
$$ J(W) = ||Y - \hat{Y}||^2_2 = ||Y - XW||^2_2 = \sum_{i=1}^n (y_i - x_i^T W)^2 $$

**Mathematical Derivation**:
To find the optimal $W$ that minimizes $J(W)$, I take the gradient with respect to $W$ and set it to zero (convex optimization):

1.  **Expand the term**:
    $$ J(W) = (Y - XW)^T (Y - XW) = Y^TY - 2W^TX^TY + W^TX^TXW $$
2.  **Compute the Gradient** $\nabla_W J(W)$:
    $$ \frac{\partial J(W)}{\partial W} = -2X^TY + 2X^TXW $$
3.  **Set to Zero**:
    $$ -2X^TY + 2X^TXW = 0 \implies X^TXW = X^TY $$

This yields the **Normal Equation**:
$$ W = (X^T X)^{-1} X^T Y $$

**NumPy Implementation Details**:
While the analytical solution involves the matrix inverse $(X^T X)^{-1}$, calculating it directly is computationally expensive ($O(n^3)$) and numerically unstable when $X$ has multicollinearity (making $X^TX$ close to singular).

In implementation, use **Singular Value Decomposition (SVD)** via `np.linalg.lstsq`.
-   Decompose $X = U \Sigma V^T$.
-   Compute the pseudo-inverse $X^+ = V \Sigma^+ U^T$.
-   Solve $W = X^+ Y$.
This approach minimizes the L2-norm of the solution vector, providing a robust solution even for rank-deficient matrices.

**Toy Example**:
Suppose I have 3 data points $(x, y)$: $(1, 2), (2, 3), (3, 5)$.
I want to fit $y = w_0 + w_1 x$.

1.  **Construct Matrices**:
    $$
    X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad Y = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}
    $$
    *(Note: First column of X is 1s for the intercept)*

2.  **Compute $X^T X$**:
    $$
    X^T X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}
    $$

3.  **Compute $X^T Y$**:
    $$
    X^T Y = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}
    $$

4.  **Solve Normal Equation** $(X^T X) W = X^T Y$:
    $$
    \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix} \begin{bmatrix} w_0 \\ w_1 \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}
    $$
    Solving this system yields $w_0 = -1/3, w_1 = 11/6$.
    Equation: $y = -0.33 + 1.83x$.

### 5.3. Model 2: Lasso Regression
**Objective**: Minimize SSE + L1 Penalty to induce sparsity (feature selection).
$$ J(W) = \frac{1}{2n} ||Y - XW||^2_2 + \lambda ||W||_1 $$

Lasso introduces a regularization parameter $\lambda$ that penalizes the absolute magnitude of the regression coefficients. This encourages simple, sparse models by determining which features are truly important.

### 5.4. Hyperparameter Tuning (Grid Search & K-Fold CV)
Lasso performance heavily depends on the regularization strength $\lambda$. To find the optimal $\lambda$, I implemented **K-Fold Cross-Validation (K=5)** from scratch:

**1. Data Splitting (Fancy Indexing)**:
I divide the training set indices into $k$ buckets. For each iteration $i \in [0, k-1]$:
-   **Validation Set**: `indices[i*fold_size : (i+1)*fold_size]`
-   **Training Set**: `np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])`

**2. Grid Search Loop**:
-   Define a hyperparameter grid $\Lambda = [0.0001, 0.001, 0.01, 0.1, 1.0]$.
-   For each $\lambda \in \Lambda$:
    -   Train Lasso on Training Set.
    -   Predict on Validation Set.
    -   Compute average MSE across all $k$ folds.
-   Select $\lambda^*$ that minimizes the average Validation MSE.

### 5.5. Model Evaluation Metrics
I implemented the following metrics using pure NumPy vectorization:

**1. Mean Squared Error (MSE)**
The average squared difference between the estimated values and the actual value.
$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
*NumPy Code*: `np.mean((Y - Y_pred) ** 2)`

**2. R-squared ($R^2$ Score)**
Represents the proportion of variance for a dependent variable that's explained by an independent variable.
$$ R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
*NumPy Code*: `1 - (np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y))**2))`

**3. Root Mean Squared Error (RMSE)**
Calculated on the *original scale* (after exponentiating the log-predictions) to give a dollar-value error interpretation.
$$ \text{RMSE} = \sqrt{\text{MSE}(e^Y-1, e^{\hat{Y}}-1)} $$

## 6. NumPy Techniques Used
This project showcases advanced NumPy capabilities:

-   **Structure Arrays (`np.dtype`)**: Used to load mixed-type CSV data (integers, strings, floats) efficiently into a single array, emulating a DataFrame.
-   **Vectorization**: Loops were strictly avoided for data processing.
    -   *Example*: `np.where(arr == '', '0', arr)` for cleaning.
    -   *Example*: `data['price_log'] = np.log1p(data['price'])` for transformation.
-   **Broadcasting**: Used in One-Hot Encoding and calculating residuals ($Y - XW$) without manual iteration.
-   **Boolean Masking**: Used for filtering outliers (`data[mask]`) and "loc"-like operations.
-   **Fancy Indexing**: Used in K-Fold Cross-Validation to split data (`indices[val_start:val_end]`).

## 7. Installation & Setup
1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```
2.  **Create Environment**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Core dependencies*: `numpy`, `matplotlib`, `seaborn`. (`scikit-learn` is used only for verification).

## 8. Usage
Run the notebooks in the following order to reproduce the analysis:

1.  **`01_data_exploration.ipynb`**:
    -   **Library Import**: Loads necessary libraries (`numpy`, `pandas` solely for initial peek, `matplotlib`, `seaborn`).
    -   **Data Loading**: Reads `AB_NYC_2019.csv` into a Pandas DataFrame for easy visualization.
    -   **Univariate Analysis**: Plots histograms for `price` (showing skewness) and other numerical features.
    -   **Bivariate Analysis**:
        -   Scatter plots: `price` vs `reviews`, `price` vs `availability`.
        -   Box plots: `price` distribution across `neighbourhood_group` and `room_type`.
    -   **Heatmap**: Computes and displays the Correlation Matrix to identify potential predictors.
    -   **Geospatial Plot**: Visualizes listings on a map of NYC using latitude/longitude, color-coded by price.

2.  **`02_preprocessing.ipynb`** (NumPy Pipeline):
    -   **Structured Array Loading**: Reads raw CSV using `csv` module and converts it into a NumPy Structured Array with custom dtypes (e.g., `U100`, `int32`, `float64`).
    -   **Missing Value Imputation**:
        -   Fills `reviews_per_month` NaNs with 0.
        -   Handles empty strings in categorical columns.
    -   **Outlier Removal**: Calculates IQR for log-price and removes extreme outliers.
    -   **Feature Engineering**:
        -   **Log-Transform**: Applies `np.log1p` to `price`.
        -   **Interaction Terms**: Creates `interaction_nights_reviews`.
        -   **Spatial Feature**: Creates `is_high_value_core`.
    -   **One-Hot Encoding**: Manually implements one-hot logic for `neighbourhood_group` and `room_type`, creating binary columns.
    -   **Standard Scaling**: Manually calculates $\mu$ and $\sigma$ for each feature and applies Z-score normalization.
    -   **Saving**: Exports the processed matrix to `data/processed/airbnb_processed.csv`.

3.  **`03_modeling.ipynb`** (Model Training & Evaluation):
    -   **Data Splitting**: Shuffles data indices and splits into 80% Train and 20% Test sets.
    -   **Model Implementation**:
        -   Defines `train_linear_regression(X, Y)` using `np.linalg.lstsq`.
        -   Defines `train_lasso_regression(X, Y)` using Coordinate Descent loop and soft-thresholding function.
    -   **Hyperparameter Tuning**:
        -   Implements `k_fold_cross_validation` function.
        -   Runs `grid_search` to find best $\lambda$ for Lasso.
    -   **Final Evaluation**:
        -   Trains optimized models on the full training set.
        -   Predicts on the hold-out test set.
        -   Calculates MSE, R2, and RMSE.
    -   **Verification**: Imports `sklearn.linear_model` to train reference models and compares coefficients/metrics to validate the custom NumPy implementation accuracy.

## 9. Results & Analysis
The models were evaluated on an independent 20% test set.

| Model | MSE (Log Scale) | R² Score | RMSE (Original $) |
|-------|-----------------|----------|-------------------|
| **OLS (Custom)** | 0.1927 | 0.5319 | $80.38 |
| **Lasso (Custom)** | 0.1988 | 0.5172 | $81.47 |

### Why Lasso?
Lasso Regression (Least Absolute Shrinkage and Selection Operator) was chosen as a benchmark comparison because of its unique ability to perform **feature selection**. By adding an L1 penalty term ($\lambda ||W||_1$), Lasso forces the coefficients of irrelevant or redundant features to become exactly zero.

Comparing OLS (which uses all features) against Lasso allows us to test the **quality of our feature engineering**:
-   If Lasso significantly outperforms OLS, it implies our dataset contains many noisy or irrelevant features that OLS is overfitting to.
-   If OLS performs similarly to Lasso (as seen above), it suggests that **the selected features are robust and highly relevant**.

### Conclusion on Feature Engineering
The similar performance between the unregularized OLS and the L1-regularized Lasso indicates that **the feature engineering process was highly effective**. 
1.  **High Signal-to-Noise Ratio**: We successfully filtered out noise during preprocessing (e.g., through outlier removal and careful selection of interaction terms), leaving OLS with a clean set of predictors.
2.  **No Significant Overfitting**: Despite OLS having no penalty term, it did not overfit the training data, further proving that the dimensionality of the feature space was well-managed relative to the number of samples ($n \gg p$).

Measurements confirm that the custom OLS implementation effectively captured the underlying pricing trends without needing the sparsity constraints of Lasso.

## 10. Project Structure
```
project/
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── data/
│   ├── raw/                # Original AB_NYC_2019.csv
│   └── processed/          # Processed NumPy-ready CSV
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
└── src/                    # (Optional)
```

## 11. Challenges & Solutions
1.  **Loading Mixed Data**: NumPy standard arrays require a single type.
    -   *Solution*: Used **Nx1 Structured Arrays** with a custom `dtype` to handle strings and numbers side-by-side, preserving memory efficiency.
2.  **Numerical Stability**: Calculating $(X^T X)^{-1}$ resulted in singular matrix errors due to multicollinearity.
    -   *Solution*: Switched to `np.linalg.lstsq` (SVD-based) and introduced L1 regularization (Lasso) to handle correlated features.
3.  **Lasso Convergence**: Subgradient descent oscillates if not tuned.
    -   *Solution*: Adopted the **Coordinate Descent** algorithm with Soft Thresholding, which is faster and more stable for L1 problems.

## 12. Future Improvements
-   Implement **Ridge Regression (L2)** and **ElasticNet** for comparison.
-   Add **Polynomial Features** (degree=2) to capture more non-linear interactions.
-   Implement **Mini-Batch Gradient Descent** to handle datasets larger than memory.

## 13. Contributors
-   **Author**: [Your Name/MSSV]
-   **Institution**: VNUHCM - University of Science
-   **Course**: CSC17104 - Programming for Data Science

## 14. License
This project is licensed under the MIT License.

## Assessment Criteria (Reference)
1.  **Notebook Presentation (10%)**: Clear structure, detailed explanations, and visualizations.
2.  **GitHub Repository (10%)**: Quality README, logical structure.
3.  **NumPy Techniques (50%)**:
    -   Vectorization (No for loops).
    -   Broadcasting & Ufuncs.
    -   Fancy Indexing & Masking.
    -   Mathematical correctness and code efficiency.
4.  **Analysis & Results (30%)**: Model performance, insights, and visualizations.