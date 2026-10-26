# Hyperparameter Tuning Analysis
## 1. Baseline Model
Model:
DecisionTreeClassifier
CV F1 Macro:
0.9328
Test Accuracy:
0.9333
## 2. Grid Search
Model:
RandomForestClassifier
Total combinations:
72
Cross-validation:
5-fold
Total fits:
360
Best CV F1 Macro:
0.9583
Test Accuracy:
0.9667
## 3. Random Search
Model:
RandomForestClassifier
Number of iterations:
30
Cross-validation:
5-fold
Total fits:
150
Best CV F1 Macro:
0.9581
Test Accuracy:
0.9667
## 4. Comparison
Grid Search evaluates all combinations in the specified
hyperparameter grid.
Random Search evaluates a selected number of combinations
from the search space.
In this experiment, Grid Search required 360 model fits,
while Random Search required 150 model fits.
Both tuning approaches produced higher CV F1 Macro
than the baseline model.
