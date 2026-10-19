# Setup
setwd("E:\\R.Project\\Heart")
heart = read.csv("heart.csv")
heart = unique(heart) # remove duplicates
library(dplyr)
library(tree)
library(randomForest)
library(gbm)
library(pROC)
library(pdp)
library(xgboost)
library(ParBayesianOptimization)
library(doParallel)
library(shapviz)
n = nrow(heart)
p = ncol(heart) - 1
na.counts = colSums(is.na(heart))
total.na = sum(na.counts)


# Data Dictionary
coln.heart = colnames(heart)
description = c("age in years",
                "1 = male; 0 = female",
                "chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic (no chest pain))",
                "resting blood pressure (in mm Hg on admission to the hospital)",
                "serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
                "resting electrocardiographic results",
                "maximum heart rate achieved",
                "exercise induced angina (1 = yes; 0 = no)",
                "ST depression induced by exercise relative to rest",
                "the slope of the peak exercise ST segment","number of major vessels (0-4) colored by fluoroscopy",
                "thalassemia (0 = no record; 1 = normal; 2 = fixed defect; 3 = reversible defect)",
                "the presence of heart disease in the patient (1 = yes; 0 = no)")

data.type = c("numerical",
              "categorical",
              "categorical",
              "numerical",
              "numerical",
              "categorical",
              "categorical",
              "numerical",
              "categorical",
              "numerical",
              "categorical",
              "numerical",
              "categorical",
              "categorical")

data.dictionary = data.frame("feature" = coln.heart, description, "data type" = data.type)

# Make categorical features as factor for modeling
cat.col = data.dictionary[data.dictionary$data.type == "categorical",]$feature
heart = heart %>%
  mutate(across(all_of(cat.col), as.factor))

# Train-test Split (80/20)
set.seed(325)
train.idx = sample(n, n*0.8)
train.heart = heart[train.idx,]
test.heart = heart[-train.idx,]
n.train = nrow(train.heart)
n.test = nrow(test.heart)

# EDA
## Summary Statistics
heart.table = table(train.heart$target)
output.summary = data.frame(
  "HeartDiseaseYes" = heart.table[2],
  "HeartDiseaseNo" = heart.table[1],
  "NA" = nrow(train.heart[is.na(heart$target),]),
  "Mode" = names(heart.table)[which.max(heart.table)]
)
rownames(output.summary) = "the presence of heart disease"
heart.nfeature.summary = train.heart %>%
  summarize(mean = c(mean(age),
                     mean(trestbps),
                     mean(chol),
                     mean(thalach),
                     mean(oldpeak),
                     mean(ca)),
            median = c(median(age),
                       median(trestbps),
                       median(chol),
                       median(thalach),
                       median(oldpeak),
                       median(ca)),
            std.dev = c(sd(age),
                        sd(trestbps),
                        sd(chol),
                        sd(thalach),
                        sd(oldpeak),
                        sd(ca)),
            maximum = c(max(age),
                        max(trestbps),
                        max(chol),
                        max(thalach),
                        max(oldpeak),
                        max(ca)),
            minimum = c(min(age),
                        min(trestbps),
                        min(chol),
                        min(thalach),
                        min(oldpeak),
                        min(ca)),
            range = maximum - minimum
  )
rownames(heart.nfeature.summary) = c("age", "trestbps", "chol", "thalach", "oldpeak", "ca")

## Exploratory Plots and Statistical Testings
num.col = row.names(heart.nfeature.summary)
for (i in 1:length(num.col)){
  hist(train.heart[, num.col[i]],
       xlab = num.col[i],
       freq = FALSE,
       main = "")}

for (i in 1:length(cat.col)){
  plot(train.heart[, cat.col[i]],
       xlab = cat.col[i],
       ylim = c(0, 200))}

boxplot(train.heart$ca ~ train.heart$target,
        xlab = "Heart Disease Presence
        (0 = No, 1 = Yes)",
        ylab = "The Number of Colored Vessels by fluoroscopy")
boxplot(train.heart$oldpeak ~ train.heart$target,
        xlab = "Heart Disease Presence
        (0 = No, 1 = Yes)",
        ylab = "ST Depression Induced By Exercise")
boxplot(train.heart$age ~ train.heart$target,
        xlab = "Heart Disease Presence (0 = No, 1 = Yes)",
        ylab = "Age")
boxplot(train.heart$trestbps ~ train.heart$target,
        xlab = "Heart Disease Presence (0 = No, 1 = Yes)",
        ylab = "Resting Blood Pressure")
boxplot(train.heart$chol ~ train.heart$target,
        xlab = "Heart Disease Presence (0 = No, 1 = Yes)",
        ylab = "Serum Cholesterol")
boxplot(train.heart$thalach ~ train.heart$target,
        xlab = "Heart Disease Presence (0 = No, 1 = Yes)",
        ylab = "Maximum Heart Rate")
wilcox.test(train.heart$ca ~ train.heart$target)
wilcox.test(train.heart$oldpeak ~ train.heart$target)
t.test(train.heart$age ~ train.heart$target)
wilcox.test(train.heart$trestbps ~ train.heart$target)
wilcox.test(train.heart$chol ~ train.heart$target)
wilcox.test(train.heart$thalach ~ train.heart$target)

cp.table = table("Chest Pain Type" = train.heart$cp,
                 "Presence of Heart Disease" = train.heart$target)
thal.table = table("Thalassemia" = train.heart$thal,
                   "Presence of Heart Disease" = train.heart$target)
sex.table = table("Sex" = train.heart$sex,
                  "Presence of Heart Disease" = train.heart$target)
slope.table = table("Slope" = train.heart$slope,
                    "Presence of Heart Disease" = train.heart$target)
restecg.table = table("Resting Electrocardiographic Results" = train.heart$restecg,
                      "Presence of Heart Disease" = train.heart$target)
angina.table = table("Exercise Induced Angina?" = train.heart$exang,"Heart Disease?" = train.heart$target)
barplot(cp.table,
        xlab = "Heart Disease Presence
        (0 = No, 1 = Yes)",
        main = "Chest Pain Type vs. Heart Disease Status")
legend("topright",
       inset = c(-0.3, 0),
       legend = c("typical angina",
                  "atypical angina",
                  "non-anginal pain",
                  "asymptomatic"),
       fill = gray.colors(4),
       cex = 0.7)

barplot(angina.table, 
        beside = TRUE,
        main = "Exercise Induced Angina vs. Heart Disease Status",
        xlab = "Heart Disease Presence
        (0 = No, 1 = Yes)",
        ylab = "Number of Patients")
legend("topright",
       inset = c(-0.2, 0),
       legend = c("No Angina", "Angina"),
       fill = gray.colors(4))

chisq.test(cp.table)
fisher.test(thal.table)
chisq.test(sex.table)
chisq.test(slope.table)
fisher.test(restecg.table)
chisq.test(angina.table)

plot(train.heart$trestbps, train.heart$thalach,
     col = train.heart$target,
     pch = 19,
     xlab = "Resting Blood Pressure (in mm Hg)",
     ylab = "Maximum Heart Rate Achieved",
     main = "Resting BP vs Max. Heart Rate")
plot(train.heart$oldpeak, train.heart$chol,
     col = train.heart$target,
     pch = 19,
     xlab = "ST Depression Induced by Exercise",
     ylab = "Serum Cholestoral in mg/dl",
     main = "ST Depression vs Serum Cholestoral")
plot(train.heart$age, train.heart$chol,
     col = train.heart$target,
     pch = 19,
     xlab = "Age",
     ylab = "Serum Cholestoral in mg/dl",
     main = "Age vs Max. Serum Cholestoral")

# Fit a Plain Tree
set.seed(325)
heart.tree = tree(as.factor(target) ~ ., data = train.heart,
                  mindev=0,
                  minsize=1)

## Use CV to pick the best size
cv.heart = cv.tree(heart.tree, FUN = prune.misclass)
min.dev = min(cv.heart$dev)
best.size = min(cv.heart$size[cv.heart$dev == min.dev])

## Pruning
heart.tree.pruned = prune.misclass(heart.tree, best = best.size)

## ROC
heart.pruned.prob = predict(heart.tree.pruned, test.heart, type = "vector")
heart.pruned.prob = heart.pruned.prob[,2]
roc.pruned = roc(test.heart$target, heart.pruned.prob)
plot(roc.pruned,
     col = "red",
     main = "ROC Curve: Pruned Decision Tree",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = TRUE)
best.threshold.tree = coords(roc.pruned, "best")

## Confusion Matrix, Test Error and Model Performance
heart.pruned.class = ifelse(heart.pruned.prob > best.threshold.tree$threshold,
                            1, 0)
confusion.pruned = table("actual" = test.heart$target,
                  "predicted" = heart.pruned.class)
test.error.pruned = (confusion.pruned[1,2] + confusion.pruned[2,1])/n.test
fpr.pruned = confusion.pruned[1,2]/(confusion.pruned[1,1] + confusion.pruned[1,2])
fnr.pruned = confusion.pruned[2,1]/(confusion.pruned[2,1] + confusion.pruned[2,2])
tpr.pruned = confusion.pruned[2,2]/(confusion.pruned[2,1] + confusion.pruned[2,2])
tnr.pruned = confusion.pruned[1,1]/(confusion.pruned[1,1] + confusion.pruned[1,2])

## Plain Tree Interpretation
plot(heart.tree.pruned)
text(heart.tree.pruned)

# Random Forest
## Split X and Y for training and test set
x.train = train.heart %>%
  select(-target)
y.train = as.factor(train.heart$target)
x.test = test.heart %>% 
  select(-target)
y.test = as.factor(test.heart$target)

## Tuning Parameters
set.seed(325)
heart.rf.tune = tuneRF(x.train, y.train,
                       stepFactor = 1.5,
                       improve = 0.01,
                       ntreeTry = 2000,
                       trace = TRUE,
                       plot = TRUE,
                       nodesize = 10
)

min.oob = min(heart.rf.tune[,2])
best.m = heart.rf.tune[,1][heart.rf.tune[,2] == min.oob]

## Fit a random forest
set.seed(325)
heart.rf = randomForest(x.train, y.train, 
                        ntree = 2000,
                        mtry = best.m,
                        nodesize = 10)

## ROC
heart.rf.prob = predict(heart.rf, x.test, type = "vote")
heart.rf.prob = heart.rf.prob[,2]
roc.rf = roc(test.heart$target, heart.rf.prob)
plot(roc.rf,
     col = "red",
     main = "ROC Curve: Random Forest",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = TRUE)
best.threshold.rf = coords(roc.rf, "best")

## Confusion Matrix, Test Error and Model Performance
heart.rf.class = ifelse(heart.rf.prob > best.threshold.rf$threshold,
                        1, 0)
confusion.rf = table("actual" = test.heart$target,
                     "predicted" = heart.rf.class)
test.error.rf = (confusion.rf[1,2] + confusion.rf[2,1])/n.test
fpr.rf = confusion.rf[1,2]/(confusion.rf[1,1] + confusion.rf[1,2])
fnr.rf = confusion.rf[2,1]/(confusion.rf[2,1] + confusion.rf[2,2])
tpr.rf = confusion.rf[2,2]/(confusion.rf[2,1] + confusion.rf[2,2])
tnr.rf = confusion.rf[1,1]/(confusion.rf[1,1] + confusion.rf[1,2])

## Random Forest Interpretation
### Variable Importance
rf.importance = data.frame(heart.rf$importance)
rf.importance = sort_by(rf.importance, rf.importance$MeanDecreaseGini, 
                        decreasing = TRUE)
rf.importance = cbind("feature" = rownames(rf.importance), rf.importance)
rownames(rf.importance) = NULL
top3.var.rf = head(rf.importance$feature, 3)
rf.importance$feature = factor(rf.importance$feature, 
                               levels = rf.importance$feature)
par(mar = c(5, 5, 4, 2))
barplot(rev(rf.importance$MeanDecreaseGini),
        names.arg = rev(rf.importance$feature), 
        horiz = TRUE,
        las = 1,
        xlim = c(0,20),
        xlab = "Mean Decrease Gini",
        main = "Variable Importance From Random Forest"
)

## Partial Dependence Plot for Top 3 features
par(mfrow = c(1,3))
for (i in 1:length(top3.var.rf)){
  partialPlot(heart.rf, x.train, top3.var.rf[i],
              xlab = top3.var.rf[i],
              main = "",
              which.class = "1")
}

# Boosting Trees
train.heart$target = as.numeric(train.heart$target)
train.heart$target = ifelse(train.heart$target == 2, 1, 0)

## Tuning a Boosted Tree
boost.grid = expand.grid(
  shrinkage = c(0.001, 0.01, 0.015),
  interaction.depth = c(1, 2),
  n.minobsinnode = c(10, 20),
  optimal.trees = 0,
  min.cv.deviance = 0
)

id = NULL
learning.rate = NULL
min.obs.node = NULL

for (i in 1:nrow(boost.grid)){
  set.seed(325)
    boost.temp = gbm(target ~ .,data = train.heart,
                     distribution = "bernoulli",
                     n.trees = 5000,
                     interaction.depth = boost.grid$interaction.depth[i],
                     shrinkage = boost.grid$shrinkage[i],
                     n.minobsinnode = boost.grid$n.minobsinnode[i],
                     n.cores = 6,
                     cv.folds = 5)
    boost.grid$optimal_trees[i] = which.min(boost.temp$cv.error)
    boost.grid$min.cv.deviance[i] = min(boost.temp$cv.error)
    id = append(id, boost.grid$interaction.depth[i])
    learning.rate = append(learning.rate, boost.grid$shrinkage[i])
    min.obs.node = append(min.obs.node, boost.grid$n.minobsinnode[i])
    
}

tuning.df = data.frame("interaction depth" =id, "learning rate" = learning.rate,
                       "min obs in node" = min.obs.node, 
                       "optimal tree" = boost.grid$optimal_trees,
                       "cv deviance" = boost.grid$min.cv.deviance)
best.tuning = tuning.df[which.min(tuning.df$cv.deviance),]
# interaction.depth = 2
# learning.rate = 0.001
# min.obs.in.node = 20
# optimal.tree = 4964
# cv.deviance = 0.7833541

## Fit a Boosted Tree
set.seed(325)
heart.boost = gbm(target ~ .,data = train.heart,
                  distribution = "bernoulli",
                  n.trees = best.tuning$optimal.tree,
                  interaction.depth = best.tuning$interaction.depth,
                  shrinkage = best.tuning$learning.rate,
                  n.minobsinnode = best.tuning$min.obs.in.node,
                  n.cores = 6)


## ROC
heart.boost.prob = predict(heart.boost, test.heart, type = "response")
roc.boost = roc(test.heart$target, heart.boost.prob)
plot(roc.boost,
     col = "red",
     main = "ROC Curve: Boosted Tree",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = TRUE)
best.threshold.boost = coords(roc.boost, "best")

## Confusion Matrix, Test Error and Model Performance
heart.boost.class = ifelse(heart.boost.prob > best.threshold.boost$threshold,
                           1, 0)
confusion.boost = table("actual" = test.heart$target,
                        "predicted" = heart.boost.class)
test.error.boost = (confusion.boost[1,2] + confusion.boost[2,1])/n.test
fpr.boost = confusion.boost[1,2]/(confusion.boost[1,1] + confusion.boost[1,2])
fnr.boost = confusion.boost[2,1]/(confusion.boost[2,1] + confusion.boost[2,2])
tpr.boost = confusion.boost[2,2]/(confusion.boost[2,1] + confusion.boost[2,2])
tnr.boost = confusion.boost[1,1]/(confusion.boost[1,1] + confusion.boost[1,2])

## Variable Importance
boost.importance = summary(heart.boost, las = 1, 
                           main = "Variable Importance From Boosted Tree")
top3.var.boost = head(boost.importance$var, 3)

## Partial Dependence Plot for Top 3 features
pd = list()
for (i in 1:length(top3.var.boost)){
  pd[[i]] = partial(heart.boost,
               pred.var = top3.var.boost[i],
               n.trees = best.tuning$optimal.tree,
               train = train.heart,
               type = "classification")
}
par(mfrow = c(1, 3))
barplot(pd[[1]]$yhat, 
        names.arg = pd[[1]]$thal, 
        col = "blue",
        main = "",
        ylab = "Log-Odds",
        xlab = "thal")
plot(pd[[2]]$yhat,
     type = "l",
     ylab = "Log-Odds",
     xlab = "ca")
barplot(pd[[3]]$yhat, 
        names.arg = pd[[3]]$cp, 
        col = "blue",
        main = "",
        ylab = "Log-Odds",
        xlab = "cp")

# XGBoost
## Setup
y.train = as.numeric(y.train)
y.train = ifelse(y.train == 2, 1, 0)
y.test = as.numeric(y.test)
y.test = ifelse(y.test == 2, 1, 0)
dtrain = xgb.DMatrix(data = x.train, label = y.train)
dtest = xgb.DMatrix(data = x.test, label = y.test)

## Tuning
scoring.func = function(eta,
                        max_depth,
                        min_child_weight,
                        subsample,
                        colsample_bytree,
                        lambda,
                        alpha,
                        gamma){
  dtrain = xgb.DMatrix(data = x.train, label = y.train)
  
  params = list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    lambda = lambda,
    alpha = alpha,
    gamma = gamma
  )
  
  xgbcv = xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1500,
    nfold = 5,
    showsd = FALSE,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  return(list(Score  = max(xgbcv$evaluation_log$test_auc_mean),
              nrounds = which.max(xgbcv$evaluation_log$test_auc_mean)))
}

bounds = list(
  eta = c(0.005, 0.2),
  max_depth = c(2L, 10L),
  min_child_weight = c(1, 10),
  subsample = c(0.5, 1),
  colsample_bytree = c(0.5, 1),
  gamma = c(0, 5),
  lambda = c(0, 5),
  alpha = c(0, 5)
)

cl = makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)
clusterExport(cl, c("x.train", "y.train"))
clusterEvalQ(cl, library(xgboost))

set.seed(325)
bayes.opt = bayesOpt(
  FUN = scoring.func,
  bounds = bounds,
  initPoints = 40,
  iters.n = 80,
  iters.k = parallel::detectCores() - 1
)
stopCluster(cl)
best.param = getBestPars(bayes.opt)
best.iter = bayes.opt$scoreSummary$nrounds[which.max(bayes.opt$scoreSummary$Score)]
saveRDS(bayes.opt, "bayes_opt_result.rds")

## fit a XGBoost
params = xgb.params(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best.param$eta,
  max_depth = best.param$max_depth,
  min_child_weight = best.param$min_child_weight,
  subsample = best.param$subsample,
  colsample_bytree = best.param$colsample_bytree,
  lambda = best.param$lambda,
  alpha = best.param$alpha,
  gamma = best.param$gamma
)


set.seed(325)
heart.xgboost = xgb.train(
  params = params,
  data = dtrain,
  nrounds = best.iter
)

## ROC
heart.xgboost.prob = predict(heart.xgboost, dtest)
roc.xgboost = roc(test.heart$target, heart.xgboost.prob)
plot(roc.xgboost,
     col = "red",
     main = "ROC Curve: XGBoost Tree",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     print.auc = TRUE)
best.threshold.xgboost = coords(roc.xgboost, "best")

## Confusion Matrix, Test Error and Model Performance
heart.xgboost.class = ifelse(heart.xgboost.prob > best.threshold.xgboost$threshold,
                           1, 0)
confusion.xgboost = table("actual" = test.heart$target,
                        "predicted" = heart.xgboost.class)
test.error.xgboost = (confusion.xgboost[1,2] + confusion.xgboost[2,1])/n.test
fpr.xgboost = confusion.xgboost[1,2]/(confusion.xgboost[1,1] + confusion.xgboost[1,2])
fnr.xgboost = confusion.xgboost[2,1]/(confusion.xgboost[2,1] + confusion.xgboost[2,2])
tpr.xgboost = confusion.xgboost[2,2]/(confusion.xgboost[2,1] + confusion.xgboost[2,2])
tnr.xgboost = confusion.xgboost[1,1]/(confusion.xgboost[1,1] + confusion.xgboost[1,2])

## Variable Importance
xgboost.importance = xgb.importance(model = heart.xgboost)
xgb.plot.importance(xgboost.importance)

## Shapley Values
shap = shapviz(heart.xgboost, 
               X_pred = xgb.DMatrix(data = x.train),
               X = x.train)

sv_importance(shap, kind = "beeswarm")
