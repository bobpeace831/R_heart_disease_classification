# Heart Disease Classification with Tree-Based Models

Predicting the presence of heart disease from 13 clinical features. The project compares four tree-based classifiers, from a single pruned tree to Bayesian-optimized XGBoost, and uses variable importance, partial dependence plots, and SHAP values to see which clinical signals drive the predictions.

![R](https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB5E28)
![Task](https://img.shields.io/badge/task-binary%20classification-blue)

> ⚠️ For educational and portfolio purposes only. Not intended for clinical diagnosis.

---

## Dataset

`heart.csv` is the widely used heart disease dataset derived from the UCI Heart Disease (Cleveland) data. Duplicate rows are removed before modeling.

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
- Summary statistics for numerical features and the class balance of `target`
- Histograms, bar plots, boxplots by class, and scatterplots colored by class
- **Statistical tests** comparing patients with and without heart disease:
  - Numerical features: Wilcoxon rank-sum tests (`ca`, `oldpeak`, `trestbps`, `chol`, `thalach`) and a t-test (`age`)
  - Categorical features: chi-squared tests (`cp`, `sex`, `slope`, `exang`) and Fisher's exact tests (`thal`, `restecg`)

### 2. Models
80/20 train–test split (`set.seed(325)`).

| Model | Package | Tuning |
|---|---|---|
| Pruned decision tree | `tree` | Grown fully, pruned to the CV-optimal size by misclassification (`cv.tree`) |
| Random forest | `randomForest` | `mtry` chosen by OOB error (`tuneRF`), 2,000 trees, `nodesize = 10` |
| Gradient boosting | `gbm` | Grid search over shrinkage, interaction depth, and min node size with 5-fold CV |
| XGBoost | `xgboost` | Bayesian optimization (`ParBayesianOptimization`) over 8 hyperparameters, 5-fold CV AUC with early stopping, run in parallel |

**Selected GBM configuration:** interaction depth = 2, learning rate = 0.001, min obs per node = 20, 4,964 trees (CV deviance ≈ 0.783).

**XGBoost search space:** `eta` [0.005, 0.2], `max_depth` [2, 10], `min_child_weight` [1, 10], `subsample` [0.5, 1], `colsample_bytree` [0.5, 1], `gamma` [0, 5], `lambda` [0, 5], `alpha` [0, 5]. 40 initial points + 80 iterations.

### 3. Evaluation
For each model on the held-out test set:
- ROC curve and AUC (`pROC`)
- Classification threshold chosen by Youden's J (`coords(roc, "best")`)
- Confusion matrix, test error, TPR (sensitivity), TNR (specificity), FPR, FNR

### 4. Interpretation
- **Pruned tree:** plotted tree structure
- **Random forest:** Mean Decrease Gini importance plus partial dependence plots for the top 3 features
- **GBM:** relative influence plus partial dependence plots (log-odds) for the top 3 features
- **XGBoost:** gain-based importance and a **SHAP beeswarm plot** (`shapviz`)

## Results

| Model | Test AUC | Test Error | Sensitivity | Specificity |
|---|---|---|---|---|
| Pruned Tree | – | – | – | – |
| Random Forest | – | – | – | – |
| GBM | – | – | – | – |
| XGBoost | – | – | – | – |

<!-- Fill in from: roc.<model>$auc, test.error.<model>, tpr.<model>, tnr.<model> -->

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

   The XGBoost Bayesian optimization uses `detectCores() - 1` workers and is the slowest step. Results are cached to `bayes_opt_result.rds`.

## Author

**Chi Hou (Bob) Lok**
