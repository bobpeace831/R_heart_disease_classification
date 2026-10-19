# Heart Disease Classification with Tree-Based Models

Predicting the presence of heart disease from 13 clinical features. The project compares four tree-based classifiers, from a single pruned tree to Bayesian-optimized XGBoost, and uses variable importance, partial dependence plots, and SHAP values to see which clinical signals drive the predictions.

![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28)
![Task](https://img.shields.io/badge/task-binary%20classification-blue)

> ⚠️ For educational and portfolio purposes only. Not intended for clinical diagnosis.

---

## Highlights

- **Best test AUC: 0.920** (XGBoost), up from 0.704 for a single pruned tree
- **Gradient boosting (GBM) is the recommended model.** It ties XGBoost on test error and has the lowest false negative rate, and it is more stable on a small sample.
- Lowering the GBM threshold to ≈ 0.26 gives **zero false negatives** on the test set, at a false positive rate of 0.33.
- **Thalassemia (`thal`), vessels colored by fluoroscopy (`ca`), and chest pain type (`cp`)** are the top 3 predictors in every ensemble model.

## Dataset

Patient-level data collected between 1981 and 1984 at hospitals in Cleveland, Hungary, Switzerland, and Long Beach ([Detrano et al., 1989](https://doi.org/10.1016/0002-9149(89)90524-9)). The 13 features are the subset Detrano et al. identified as widely available in clinical settings. All patients were already suspected of heart disease, so the classes are roughly balanced and the models are expected to have higher internal than external validity.

After removing duplicate rows, the data is split 80/20 into **241 training** and **61 test** patients. EDA and all tuning use only the training set.

| Feature | Description | Type |
|---|---|---|
| `age` | Age in years | Numerical |
| `sex` | 1 = male, 0 = female | Categorical |
| `cp` | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical |
| `restecg` | Resting electrocardiographic results | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise-induced angina | Categorical |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical |
| `slope` | Slope of the peak exercise ST segment | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy | Numerical |
| `thal` | Thalassemia (0 = no record, 1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |
| `target` | Heart disease present (1 = yes, 0 = no) | Categorical |

## Workflow

### 1. Exploratory Data Analysis
- Summary statistics, histograms, bar plots, boxplots by class, scatterplots, and a correlation heatmap
- **Statistical tests:** Wilcoxon rank-sum tests and a Welch t-test (`age`) for numerical features; chi-squared and Fisher's exact tests for categorical features
- Every feature differs significantly between the two classes (p < 0.05). Correlations among numerical features are weak (|r| ≤ 0.39), so no feature is redundant and tree-based models are a natural fit for the non-linear structure.

### 2. Models

| Model | Package | Tuning | Selected |
|---|---|---|---|
| Pruned decision tree | `tree` | 10-fold CV on tree size (misclassification) | Size 2: one split on `thal` |
| Random forest | `randomForest` | `mtry` by OOB error, 2,000 trees, `nodesize = 10` | `mtry = 3` (OOB error 18.3%) |
| Gradient boosting | `gbm` | Grid search with 5-fold CV over shrinkage, depth, and min node size | depth 2, shrinkage 0.001, min node 20, 4,964 trees |
| XGBoost | `xgboost` | Bayesian optimization over 8 hyperparameters (40 initial + 80 iterations), 5-fold CV AUC | `eta` 0.167, `max_depth` 7, 64 rounds |

<details>
<summary>Full XGBoost hyperparameters</summary>

| eta | max_depth | min_child_weight | subsample | colsample_bytree | gamma | lambda | alpha |
|---|---|---|---|---|---|---|---|
| 0.167 | 7 | 2.63 | 0.688 | 0.805 | 1.33 | 1.49 | 2.81 |

</details>

### 3. Evaluation
ROC/AUC (`pROC`), thresholds chosen by Youden's J, confusion matrices, test error, sensitivity, and specificity.

### 4. Interpretation
Plotted tree structure, Gini and relative-influence importance, partial dependence plots, XGBoost gain importance, and a SHAP beeswarm plot (`shapviz`).

## Results

Test set performance (n = 61):

| Model | Test AUC | Test Error | Sensitivity | Specificity | False Negatives |
|---|---|---|---|---|---|
| Pruned Tree | 0.704 | 0.279 | 0.784 | 0.625 | 8 |
| Random Forest | 0.909 | 0.115 | 0.919 | 0.833 | 3 |
| **GBM** | **0.917** | **0.098** | **0.946** | **0.833** | **2** |
| XGBoost | 0.920 | 0.098 | 0.946 | 0.833 | 2 |

**GBM with a clinically tuned threshold (≈ 0.26):**

| Test Error | Sensitivity | Specificity | False Negatives |
|---|---|---|---|
| 0.131 | 1.000 | 0.667 | 0 |

### Key Findings

1. **The ensembles beat a single tree by a wide margin.** AUC rises from 0.70 to above 0.90, because aggregation reduces the pruned tree's high variance.
2. **First- and second-order boosting perform about the same.** GBM and XGBoost produce identical confusion matrices. On a sample this small, XGBoost's Hessian information adds little.
3. **GBM is recommended over XGBoost.** XGBoost converged at a high learning rate after only 64 rounds, which raises stability concerns. GBM learns gradually over thousands of trees.
4. **False negatives cost the most in a clinical setting.** A missed diagnosis can mean no treatment, heart attack, or stroke. Lowering the threshold removes all false negatives in exchange for more false positives.
5. **The same three predictors lead every ensemble:** `thal`, `ca`, and `cp`. Fewer vessels colored by fluoroscopy means higher risk. A fixed thalassemia defect (`thal = 2`) is associated with higher predicted risk in the SHAP analysis.

## Limitations

- **Threshold selection uses the test set.** Each model's "best" threshold is chosen from its test ROC curve, so test error, sensitivity, and specificity are optimistic. AUC is unaffected. Choosing thresholds from out-of-fold or OOB training predictions would give unbiased estimates.
- **Small test set.** With 61 patients, one misclassification changes test error by about 1.6 points, so differences between the top three models are within noise.
- **Population.** All patients were already suspected of heart disease, and the data is from the 1980s. Results may not carry over to general screening or modern populations.

## Repository Structure

```
.
├── heart.R                  # Full pipeline: EDA, modeling, evaluation, interpretation
├── heart.csv                # Dataset
├── bayes_opt_result.rds     # Saved Bayesian optimization results (generated)
└── README.md
```

## Reproducing the Analysis

1. Install the required packages:

   ```r
   install.packages(c(
     "dplyr", "tree", "randomForest", "gbm", "pROC", "pdp",
     "xgboost", "ParBayesianOptimization", "doParallel", "shapviz"
   ))
   ```

   The script uses `xgb.params()`, which requires **xgboost ≥ 3.0**.

2. Place `heart.csv` in the project root and replace the hard-coded `setwd("E:\\R.Project\\Heart")` with your own path, or remove it and open the project as an RStudio project.

3. Run the script:

   ```r
   source("heart.R")
   ```

   The XGBoost Bayesian optimization uses `detectCores() - 1` workers and is the slowest step. Results are cached to `bayes_opt_result.rds`. Because it runs in parallel, XGBoost results may differ slightly between machines.

## Reference

Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., Guppy, K. H., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology, 64*(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9

## Author

**Chi Hou (Bob) Lok**
