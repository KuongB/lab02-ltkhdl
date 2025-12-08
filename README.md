### 3. 📓 02_preprocessing.ipynb - Advanced Feature Engineering

The Agent needs to enhance the feature set by creating new features based on non-linear relationships and interactions. This step must be integrated after initial standardization.

#### 3.4. Advanced Feature Creation (NumPy ONLY) 💡

1.  **Interaction Features (Tương quan):** Create at least **two** new interaction features by multiplying existing, standardized numerical features to capture dependency between them.
    * *Example:* $\text{latitude} \times \text{longitude}$ or $\text{reviews\_per\_month} \times \text{minimum\_nights}$.
2.  **Polynomial Features (Bậc 2):** Create at least **one** new feature by taking the square ($\text{feature}^2$) of a significant standardized numerical feature (e.g., $\text{minimum\_nights\_std}^2$) to capture non-linear effects.
3.  **Feature Matrix Update:** Recreate the **Full Feature Matrix** ($X_{\text{full}}$) and the **Reduced Feature Matrix** ($X_{\text{reduced}}$) to include these new interaction and polynomial features. This new $X_{\text{full}}$ will be used for all subsequent modeling in Notebook 03.

---

### 4. 📓 03_modeling.ipynb - Introduce Model 3

The Agent must implement a third, non-linear model (KNN) from scratch for comprehensive comparison.

#### 4.1. Setup and Splitting (Updated)
[The Agent will use the new $X_{\text{full}}$ and $X_{\text{reduced}}$ which contain the advanced features.]

#### 4.2. Model 1: Reduced Feature Linear Regression (Recalculate)
[Recalculate Model 1 performance using the new feature sets.]

#### 4.3. Model 2: Full Feature Linear Regression (Stabilized & Recalculate)
[Recalculate Stabilized Model 2 performance using the new feature sets.]

#### 4.5. Model 3: K-Nearest Neighbors (KNN) Regression 🎯

1.  **Algorithm Selection:** Implement **K-Nearest Neighbors (KNN) Regression** from scratch using NumPy.
2.  **Implementation (NumPy ONLY):**
    * Implement a distance calculation function (e.g., Euclidean distance using `np.sqrt` and `np.sum`).
    * Implement the core KNN prediction logic: find $K=5$ (or another default value) nearest neighbors and return the **mean** of their $Y$ values as the prediction.
3.  **Cross-Validation:** Run **Custom 5-Fold Cross-Validation** for Model 3 (KNN) using the **Full Feature Set ($X_{\text{full}}$)**.
4.  **Evaluation:** Calculate and report the **Avg MSE** and **Avg $R^2$ Score** for Model 3.

#### 4.6. Final Comparison and Analysis (Updated)

1.  **Compare:** Compare the final performance (MSE, $R^2$) of **Model 1 (Reduced LR)**, **Model 2 (Full Stabilized LR)**, and **Model 3 (KNN)**.
2.  **Analyze:** The analysis must discuss:
    * The impact of the **Advanced Features** (interactions/polynomials) on the $R^2$ of Model 2 compared to the previous run.
    * The comparison between