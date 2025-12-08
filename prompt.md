## prompt.md - Agent Task Instructions

The Agent is required to complete a Data Science assignment using the **New York City Airbnb Open Data** dataset. The project must be structured across three Jupyter notebooks within the `notebooks/` directory: `01_data_exploration.ipynb`, `02_preprocessing.ipynb`, and `03_modeling.ipynb`.

**STRICT TECHNICAL CONSTRAINTS:**
1.  **Data Processing & Calculation:** **Exclusively use NumPy** for all tasks, including loading data, cleaning, imputation, normalization, and statistical computation. **No Pandas, Scikit-learn (for preprocessing), or other data handling libraries.**
2.  **Visualization:** **Exclusively use Matplotlib and Seaborn.**
3.  **Modeling (Advanced):** Implement the core model (Linear Regression) and evaluation metrics **using NumPy** to achieve the highest score. Scikit-learn is an acceptable fallback for modeling only if NumPy implementation is not feasible.

---

### 1. Data Loading and Initial Inspection (Start in `01_data_exploration.ipynb`)

1.  **Data Loading (NumPy Only):** Load the `AB_NYC_2019.csv` file using `np.genfromtxt` or a combination of Python's built-in CSV reader and `np.array()` conversion. Ensure the data is loaded into a structured NumPy array or a set of corresponding NumPy arrays, handling mixed data types.
2.  **Display Data Overview (NumPy Only):**
    * Print the **first 5 rows** of the loaded data.
    * Print the **column names (features)**.
    * Print the **shape (number of rows and columns)** of the data array.

---

### 2. Data Exploration (`01_data_exploration.ipynb`)

Perform Exploratory Data Analysis (EDA) by posing and answering analytical questions.

#### 2.1. Analytical Questions and Visualization

Address the following key questions using NumPy for calculation and Matplotlib/Seaborn for visualization:

| Question | NumPy Calculation & Result | Visualization Type |
| :--- | :--- | :--- |
| **Price Distribution by Borough:** What is the average listing price across the 5 main boroughs (`neighbourhood_group`)? | Calculate **mean price** ($\mu$) per group. | **Bar Chart** comparing means. |
| **Room Type Market Share:** What is the proportion of each `room_type`? | Calculate **frequency count** and **percentage** using `np.unique()`. | **Pie Chart** showing percentages. |
| **Activity vs. Price:** Is there a correlation between `price` and the listing's popularity (`number_of_reviews`)? | Calculate **correlation coefficient** ($\rho$) using `np.corrcoef()`. | **Scatter Plot**. |
| **Busiest Hosts:** Identify the **Top 10 Host IDs** by the number of active listings (`calculated_host_listings_count`). | Count host IDs and perform descending sort. | **Horizontal Bar Chart** (Top N). |

#### 2.2. Technical EDA Requirements
* Use NumPy functions for all statistical measures (`np.mean()`, `np.std()`, `np.median()`, `np.percentile()`, `np.unique()`, etc.).
* Ensure all charts are titled and axes are labeled clearly.

---

### 3. Data Preprocessing (`02_preprocessing.ipynb`)

Prepare the data for modeling using **NumPy exclusively**.

#### 3.1. Handling Missing Values (Imputation)
1.  **Identify:** Determine columns with missing values (e.g., `reviews_per_month`, `last_review`).
2.  **Impute `reviews_per_month`:** Replace missing (NaN) values with **0** using NumPy masking or `np.nan_to_num`. (Assumption: 0 reviews if null).
3.  **Impute other numerics:** If any other essential numerical column has missing values, impute them using the **median** (`np.median()`).

#### 3.2. Outlier Treatment and Standardization/Normalization
1.  **Outlier Handling (`price`):**
    * Identify outliers using the **Interquartile Range (IQR)** method ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) using `np.percentile()`.
    * Apply a **Log Transformation** (`np.log1p()`) to the `price` column to address its right-skewed distribution, or **clip** the values to the calculated upper bound.
2.  **Feature Scaling:** Apply **Standardization (Z-score)** to the main numerical features (e.g., Log-Price, `minimum_nights`, `number_of_reviews`, `availability_365`):
    $$Z = \frac{X - \mu}{\sigma}$$
    where $\mu$ is the mean and $\sigma$ is the standard deviation (calculated via NumPy).

#### 3.3. Feature Engineering and Encoding
1.  **Feature Engineering:** Create a new meaningful numerical feature, such as **Review Density** (`reviews_per_month` / `calculated_host_listings_count`).
2.  **Categorical Encoding:** Select the most important categorical features (`neighbourhood_group`, `room_type`) and perform **One-Hot Encoding** manually using NumPy array manipulation (masking/concatenation) for integration into the linear model.

---

### 4. 📓 03_modeling.ipynb - Simple Modeling (Advanced) 🚀

Implement **TWO** Linear Regression models from scratch using NumPy for comparative analysis. The target variable ($Y$) for both models is the **normalized Log-Transformed Price**.

#### 4.1. Setup and Splitting (NumPy ONLY)
1.  **Feature/Target Preparation:**
    * Create the **Full Feature Matrix** $X_{\text{full}}$ (all preprocessed features, including engineered and encoded ones).
    * Create the **Reduced Feature Matrix** $X_{\text{reduced}}$ (select **3 to 5 key features** from $X_{\text{full}}$ based on EDA/domain knowledge).
    * Create the target vector $Y$ (normalized log-price).
2.  **Train-Test Split:** Split $X_{\text{full}}$, $X_{\text{reduced}}$, and $Y$ into training and testing sets (e.g., 80/20 ratio) using `np.random.shuffle()` and indexing.

#### 4.2. Model 1: Reduced Feature Linear Regression (For Comparison)
1.  **Model Implementation:** Implement Multivariate Linear Regression using the **Normal Equation** on the $X_{\text{reduced}}$ set:
    $$\mathbf{W}_{\text{red}} = (\mathbf{X}_{\text{reduced}}^{\text{T}}\mathbf{X}_{\text{reduced}})^{-1}\mathbf{X}_{\text{reduced}}^{\text{T}}\mathbf{Y}$$
    * Ensure the feature matrix includes the intercept term (column of ones).
2.  **Evaluation (NumPy ONLY):** Calculate the **MSE** and **$R^2$ Score** on the test set for Model 1.

#### 4.3. Model 2: Full Feature Linear Regression (Main Task)
1.  **Model Implementation:** Implement Multivariate Linear Regression using the **Normal Equation** on the $X_{\text{full}}$ set:
    $$\mathbf{W}_{\text{full}} = (\mathbf{X}_{\text{full}}^{\text{T}}\mathbf{X}_{\text{full}})^{-1}\mathbf{X}_{\text{full}}^{\text{T}}\mathbf{Y}$$
2.  **Training:** Calculate the optimal weight vector $\mathbf{W}_{\text{full}}$ using $X_{\text{full, train}}$ and $Y_{\text{train}}$.
3.  **Evaluation (NumPy ONLY):** Calculate the **MSE** and **$R^2$ Score** on the test set for Model 2.

#### 4.4. Final Comparison and Analysis
1.  **Results Reporting:** Clearly report the MSE and $R^2$ scores for both Model 1 and Model 2.
2.  **Analysis:** Compare the performance of the two models and provide an analysis/discussion on the impact of using the additional features (e.g., did the added features improve the performance significantly?).