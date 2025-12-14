# NYC Airbnb Price Prediction: A NumPy-Only Approach

## 1. Project Overview
This project focuses on analyzing and predicting Airbnb listing prices in New York City using the **2019 NYC Airbnb Open Data**. 

The core technical challenge and distinguishing feature of this project is the **exclusive use of NumPy** for the entire data pipeline—including data loading, cleaning, complex feature engineering, and the implementation of **Ordinary Least Squares (OLS) Regression** from scratch. High-level libraries like Pandas or Scikit-learn were intentionally avoided for the core logic to demonstrate a deep understanding of vectorized operations, memory management with structured arrays, and the mathematical foundations of regression algorithms. Scikit-learn is utilized solely for Lasso Regression to provide a benchmark for regularization.

## 2. Table of Contents
1. [Project Overview](#1-project-overview)
2. [Introduction](#3-introduction)
3. [Dataset & Exploratory Analysis](#4-dataset--exploratory-analysis)
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
14. [Acknowledgements](#15-acknowledgements)

## 3. Introduction
### Problem Statement
Airbnb pricing is dynamic and influenced by a multitude of factors including location, room type, availability, and host activity. Predicting an optimal price is challenging for hosts, and understanding fair value is difficult for guests.

### Motivation
By building a predictive model, we can uncover the latent drivers of listing prices. Implementing this "from scratch" allows us to peek under the hood of standard ML libraries, understanding the optimization landscapes and computational bottlenecks involved in processing real-world data.

### Objectives
- Perform Exploratory Data Analysis (EDA) to find pricing trends.
- Preprocess data using strictly **NumPy** (handling missing values, outliers, encoding).
- Implement **Linear Regression (OLS)** from mathematical first principles.
- Use **Lasso Regression** (via Scikit-learn) to analyze feature importance and regularization effects.
- Evaluate models using custom **NumPy** metrics ($MSE$, $R^2$) and compare results.

## 4. Dataset & Exploratory Analysis
- **Source**: [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
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

### 4.3. Statistical Methodology
To rigorously validate market differences, I employed **Welch’s t-test** instead of the standard Student’s t-test. This choice is crucial because real-world pricing data between different groups (Manhattan vs. Brooklyn,...) rarely satisfies the assumption of equal variances (homoscedasticity).

**Welch's t-statistic formula breakdown**:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}
$$

Where:
-   $\bar{X}$ is the sample mean.
-   $s^2$ is the unbiased sample variance (calculated via `np.var(ddof=1)`).
-   $N$ is the sample size.

**Decision Logic**:
Given our large sample size ($N > 1000$), the t-distribution converges to the Standard Normal Distribution ($Z$-distribution).
-   **Significance Level ($\alpha$)**: 0.05 (95% Confidence).
-   **Critical Value**: $\approx 1.96$.
-   **Rule**: If $|t| > 1.96$, we reject the Null Hypothesis ($H_0$), confirming the difference is statistically significant.

### 4.4. Statistical Test Results
Based on the method above, I conducted 4 key tests using pure NumPy arithmetic:

**Test 1: Location Impact (Manhattan vs. Brooklyn)**
-   **Hypothesis**:
    -   $H_0$: The average listing price in Manhattan is equal to Brooklyn ($\mu_M = \mu_B$).
    -   $H_1$: The average listing price in Manhattan is different from Brooklyn ($\mu_M \neq \mu_B$).
-   **Arithmetic Optimization**: Calculated variance and standard error using `np.var(ddof=1)` and standard arithmetic operators to avoid library overhead.
-   **Result**: T-statistic ≈ **30.48**.
-   **Conclusion**: Since $|t| > 1.96$ ($p < 0.05$), we **reject the Null Hypothesis**. There is a statistically significant price difference, confirming Manhattan is indeed the more expensive market.

**Test 2: Privacy Premium (Entire Home vs. Private Room)**
-   **Hypothesis**:
    -   $H_0$: The average listing price of "Entire home/apt" is equal to "Private room" ($\mu_{Entire} = \mu_{Private}$).
    -   $H_1$: The average prices are distinct ($\mu_{Entire} \neq \mu_{Private}$).
-   **Result**: T-statistic ≈ **58.67**.
-   **Conclusion**: With such a high t-score, we **reject the Null Hypothesis** with extreme confidence. This statistically proves that privacy is a major pricing factor, with entire homes commanding a premium of approximately $122 over private rooms.

**Test 3: Rental Duration (Short-term < 7 days vs. Long-term >= 7 days)**
-   **Hypothesis**:
    -   $H_0$: The average price of short-term rentals is equal to long-term rentals ($\mu_{Short} = \mu_{Long}$).
    -   $H_1$: The average prices are different ($\mu_{Short} \neq \mu_{Long}$).
-   **Result**: T-statistic ≈ **1.81**.
-   **Conclusion**: Since $|t| < 1.96$ ($p > 0.05$), we **FAIL to reject the Null Hypothesis**. There is statistically insufficient evidence to claim that listing price depends on the minimum stay requirement in this dataset.

**Test 4: Popularity Impact (Low Reviews <= 10 vs. High Reviews > 50)**
-   **Hypothesis**:
    -   $H_0$: The average price of low-popularity listings is equal to high-popularity listings ($\mu_{Low} = \mu_{High}$).
    -   $H_1$: The average prices are different ($\mu_{Low} \neq \mu_{High}$).
-   **Result**: T-statistic ≈ **5.96**.
-   **Conclusion**: Since $|t| > 1.96$, we **reject the Null Hypothesis**. Popular listings tend to be slightly cheaper, statistically confirming the negative correlation observed in EDA that lower prices drive higher turnover and more reviews.

## 5. Methodology & Mathematical Foundations

### 5.1. Feature Engineering (NumPy)
The preprocessing pipeline (`02_preprocessing.ipynb`) transforms raw CSV data into a clean, numerical matrix $X$:

1.  **Log-Transformation**: The target variable `price` is highly right-skewed. I apply:

    $$
    y' = \log(y + 1)
    $$

    This stabilizes variance and makes the data conform better to the Gaussian assumptions of Linear Regression.

2.  **Interaction Terms**: Capturing non-linear relationships:
    -   `interaction_nights_reviews` = `minimum_nights` $\times$ `reviews_per_month`
    -   `is_high_value_core`: A boolean mask converted to float, isolating high-density areas (Manhattan downtown) based on latitude/longitude boundaries.

3.  **One-Hot Encoding**: Categorical variables (`neighbourhood_group`, `room_type`) are converted into binary vectors using broadcasting comparisons (`data['col'][:, None] == unique_values`). The first category is dropped to prevent perfect collinearity (Dummy Variable Trap).

4.  **Standard Scaling (Z-Score)**:
    For every feature $j$:

    $$
    x_{ij}' = \frac{x_{ij} - \mu_j}{\sigma_j}
    $$

    This ensures uniform gradients during optimization.

### 5.2. Model 1: Ordinary Least Squares (OLS)
**Objective**:
Linear Regression fits a linear model with coefficients $W = (w_0, \dots, w_p)$ to minimize the Residual Sum of Squares (RSS) between the observed targets in the dataset and the targets predicted by the linear approximation.

The objective function (Cost Function) is defined as:

$$
J(W) = ||Y - \hat{Y}||^2_2 = ||Y - XW||^2_2 = \sum_{i=1}^n (y_i - x_i^T W)^2
$$

**Mathematical Derivation**:
To find the optimal $W$ that minimizes $J(W)$, I take the gradient with respect to $W$ and set it to zero (convex optimization):

1.  **Expand the term**:
    
    $$
    J(W) = (Y - XW)^T (Y - XW) = Y^TY - 2W^TX^TY + W^TX^TXW
    $$

2.  **Compute the Gradient** $\nabla_W J(W)$:
    
    $$
    \frac{\partial J(W)}{\partial W} = -2X^TY + 2X^TXW
    $$

3.  **Set to Zero**:
    
    $$
    -2X^TY + 2X^TXW = 0 \implies X^TXW = X^TY
    $$

This yields the **Normal Equation**:

$$
W = (X^T X)^{-1} X^T Y
$$

**NumPy Implementation Details**:
While the analytical solution involves the matrix inverse $(X^T X)^{-1}$, calculating it directly is computationally expensive ($O(n^3)$) and numerically unstable when $X$ has multicollinearity (making $X^TX$ close to singular).

In implementation, use **Singular Value Decomposition (SVD)** via `np.linalg.lstsq`.
-   Decompose $X = U \Sigma V^T$.
-   Compute the pseudo-inverse $X^+ = V \Sigma^+ U^T$.
-   Solve $W = X^+ Y$.
This approach minimizes the L2-norm of the solution vector, providing a robust solution even for rank-deficient matrices.

### 5.3. Model 2: Lasso Regression (Scikit-learn)
**Objective**: Minimize SSE + L1 Penalty to induce sparsity (feature selection).

$$
J(W) = \frac{1}{2n} ||Y - XW||^2_2 + \lambda ||W||_1
$$

Lasso introduces a regularization parameter $\lambda$ that penalizes the absolute magnitude of the regression coefficients. This encourages simple, sparse models by determining which features are truly important. We utilize the Scikit-learn implementation of Lasso to compare against our unregularized OLS model.

### 5.4. K-Fold Cross Validation
**Logic & Implementation**:
To robustly estimate model performance, we implemented a **K-Fold Cross-Validation** engine from scratch:

**1. Data Splitting (Fancy Indexing)**:
I divide the training set indices into $k$ buckets. For each iteration $i \in [0, k-1]$:
-   **Validation Set**: `indices[i*fold_size : (i+1)*fold_size]`
-   **Training Set**: `np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])`

**2. Loop**:
-   Train OLS on Training Set.
-   Predict on Validation Set.
-   Compute average MSE across all $k$ folds.

This custom implementation reinforces the understanding of data partitioning and model validation logic using pure NumPy.

### 5.5. Model Evaluation Metrics
I implemented the following metrics using pure NumPy vectorization:

**1. Mean Squared Error (MSE)**
The average squared difference between the estimated values and the actual value.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

*NumPy Code*: `np.mean((Y - Y_pred) ** 2)`

**2. R-squared ($R^2$ Score)**
Represents the proportion of variance for a dependent variable that's explained by an independent variable.

$$
R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

*NumPy Code*: `1 - (np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y))**2))`

**3. Root Mean Squared Error (RMSE)**
Calculated on the *original scale* (after exponentiating the log-predictions) to give a dollar-value error interpretation.

$$

\text{RMSE} = \sqrt{\text{MSE}(e^Y-1, e^{\hat{Y}}-1)}
$$

## 6. NumPy Techniques Used
This project demonstrates how to replicate a full Data Science pipeline using **pure NumPy**, effectively recreating features found in Pandas and Scikit-learn.

### 6.1. Data Structures & Logic
-   **Structured Arrays (`np.dtype`)**: emulate Pandas DataFrames by enforcing specific types (e.g., `<U100` for strings, `<f8` for floats) for each column in a single compact 1D array.
    ```python
    dtype = np.dtype([('price', np.int32), ('neighbourhood', 'U50'), ...])
    ```
-   **Vectorized Cleaning**: Replaced Python loops with `np.where` and `np.nan_to_num` for conditional logic.
    ```python
    # Example: Replace empty strings with '0' efficiently
    clean_arr = np.where(arr == '', '0', arr)
    ```

### 6.2. Advanced Indexing & Manipulation
-   **Boolean Masking**: Used for filtering rows based on complex conditions (e.g., IQR outlier removal, Geospatial filtering).
    ```python
    mask = (data['price_log'] >= lower) & (data['price_log'] <= upper)
    data_clean = data[mask]
    ```
-   **Fancy Indexing**: Used in K-Fold Cross-Validation to create non-contiguous train/validation splits.
    ```python
    val_idx = indices[start:end]
    train_idx = np.concatenate([indices[:start], indices[end:]])
    ```
-   **Broadcasting**: Key for efficient arithmetic without loops.
    -   *Standard Scaling*: `(matrix - mean_vector) / std_vector` broadcasts 1D statistics across the 2D matrix.
    -   *One-Hot Encoding*: `(column[:, None] == unique_values)` creates a boolean matrix instantly.

### 6.3. Mathematical & Linear Algebra
-   **Transformation**: `np.log1p` (natural logarithm plus 1) for stabilizing target variance.
-   **Matrix Operations**:
    -   `@` operator for matrix multiplication ($X \cdot W$).
    -   `np.column_stack` and `np.hstack` for assembling the feature matrix $X$ and adding the intercept term.
-   **Optimization**: `np.linalg.lstsq` fits the Linear Regression model using Singular Value Decomposition (SVD), ensuring numerical stability even when the matrix $X^T X$ is close to singular (non-invertible) due to multicollinearity.


## 7. Installation & Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/KuongB/lab02-ltkhdl.git
    cd lab02-ltkhdl
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
**Prerequisites**:
-   **Dataset**: The dataset is already prepared inside `data/raw/`. If not found, please download it from [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) and place `AB_NYC_2019.csv` into the directory.

Run the notebooks in the following order to reproduce the analysis:

1.  **`01_data_exploration.ipynb`**:
    -   **Pure NumPy Data Loading**: Reads the CSV file using Python's `csv` module and converts it into a NumPy array of objects, strictly avoiding Pandas.
    -   **Data Inspection**: Manually parses headers and checks data types for the first few rows to understand the structure.
    -   **Vectorized Aggregation**: Implements custom aggregation logic using `np.unique` (with `return_inverse=True`) and `np.bincount` to calculate average prices by borough without loops.
    -   **Distribution Analysis**: Uses `seaborn` (fed by NumPy arrays) to visualize the price distribution and identify the need for log-transformation.
    -   **Correlation Analysis**: Manually encodes categorical variables to compute a correlation matrix using `np.corrcoef`, visualizing relationships with a heatmap.
    -   **Geospatial Visualization**: Plots listings on a scatter plot using longitude and latitude to reveal price clusters in Manhattan and Brooklyn.

2.  **`02_preprocessing.ipynb`** (The NumPy Pipeline):
    -   **Structured Array Conversion**: Defines a explicit `np.dtype` (e.g., `U100`, `int32`, `float64`) to enforce types and enable memory-efficient storage of mixed data.
    -   **Vectorized Cleaning**:
        -   Uses `np.where` to handle empty strings and replace them with `0` or `NaT`.
        -   Fills `NaN` values in `reviews_per_month` using `np.nan_to_num`.
    -   **Outlier Removal**: Implements the IQR method on log-transformed prices using boolean masking to filter out extreme anomalies ($> 1.5 \times IQR$).
    -   **Feature Engineering**:
        -   **Log-Transform**: Applies `np.log1p` to the target `price` to normalize the distribution.
        -   **Interaction Terms**: creating `interaction_nights_reviews` (minimum_nights × reviews_per_month).
        -   **Spatial Features**: Creates `is_high_value_core` boolean feature for downtown locations.
    -   **One-Hot Encoding**: Manually implements one-hot encoding for `neighbourhood_group` and `room_type` using broadcasting comparisons, generating 0/1 float columns.
    -   **Standard Scaling**: Manually computes mean and standard deviation for numerical columns and applies Z-score normalization $(X - \mu) / \sigma$.
    -   **Export**: Saves the final processed 2D matrix to `data/processed/airbnb_processed.csv`.

3.  **`03_modeling.ipynb`** (Training & Evaluation):
    -   **Custom Data Splitting**: Implements a shuffling mechanism using `np.random.shuffle` on indices to create an 80/20 Train-Test split without `sklearn`.
    -   **OLS Implementation**:
        -   Uses the Normal Equation approach solved via SVD (`np.linalg.lstsq`) to ensure numerical stability.
        -   Manually adds an intercept column (`np.ones`) using `np.hstack`.
    -   **K-Fold Cross-Validation**:
        -   Builds a custom CV engine that slices data into $K=5$ folds using fancy indexing.
        -   Iteratively trains OLS on $K-1$ folds and validates on the remaining fold to check model robustness.
    -   **Model Evaluation**:
        -   Trains the final OLS model on the full training set.
        -   Uses **Scikit-learn's Lasso** as a benchmark for feature selection and regularization comparison.
        -   Calculates **MSE**, **RMSE** (dollar error), and **$R^2$** using custom functions built with simple NumPy array operations (`np.mean`, `np.sum`).
    -   **Verification**: Compares the custom OLS coefficients and metrics against `sklearn.linear_model.LinearRegression` to prove implementation correctness.

## 9. Results & Analysis
The models were evaluated on an independent 20% test set.

| Model | MSE (Log Scale) | R² Score | RMSE (Original $) |
|-------|-----------------|----------|-------------------|
| **OLS (Custom)** | 0.1874 | 0.5362 | $77.28 |
| **OLS (Sklearn)** | 0.1874 | 0.5362 | $77.28 |
| **Lasso (Sklearn)** | 0.1874 | 0.5362 | $77.32 |

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
└── src/                    # Optional
    ├── __init__.py
    ├── data_processing.py 
    ├── models.py           
    └── visualization.py    
```

## 11. Challenges & Solutions
1.  **Loading Mixed Data**: NumPy standard arrays require a single type.
    -   *Solution*: Used **Nx1 Structured Arrays** with a custom `dtype` to handle strings and numbers side-by-side, preserving memory efficiency.
2.  **CSV Parsing with Quoted Fields**: Using `np.genfromtxt` or `np.loadtxt` failed because some fields (like listing names) contained commas inside double quotes (e.g., "Apartment, near park"), which NumPy treated as delimiters.
    -   *Solution*: Utilized Python's built-in `csv` module to correctly parse the quoted strings first, then converted the resulting list of rows into a NumPy structured array.
3.  **Numerical Stability**: Calculating $(X^T X)^{-1}$ resulted in singular matrix errors due to multicollinearity.
    -   *Solution*: Switched to `np.linalg.lstsq` (SVD-based) and introduced L1 regularization (Lasso) to handle correlated features.
3.  **Lasso Convergence**: Subgradient descent oscillates if not tuned.
    -   *Solution*: Opted to use Scikit-learn's robust coordinate descent implementation for the Lasso benchmark to focus the "from-scratch" efforts on the core OLS and metric implementations.

## 12. Future Improvements
-   Implement **Ridge Regression (L2)** and **ElasticNet** for comparison.
-   Add **Polynomial Features** (degree>=2) to capture more non-linear interactions.
-   Implement **Mini-Batch Gradient Descent** to handle datasets larger than memory.

## 13. Contributors
-   **Author**: [Tran Tien Cuong - 23127332]
-   **Institution**: VNUHCM - University of Science
-   **Course**: CSC17104 - Programming for Data Science

## 14. License
This project is licensed under the MIT License.

## 15. Acknowledgements
I would like to express our gratitude to **Dgomonov** for providing the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) on Kaggle, which served as the foundation for this analysis.

Special thanks to the teaching staff of **CSC17104 - Programming for Data Science** at VNUHCM - University of Science for their guidance and course materials.

I also acknowledge the assistance of **Gemini 3 Pro** in correcting README syntax and refining the project content.
