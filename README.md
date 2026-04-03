# Hospital-infection-benchmark-ml
Machine learning project for predicting whether a hospital’s infection-related performance is better than the national benchmark

## The project explores the full ML workflow:

- data cleaning & preprocessing
- feature selection
- handling class imbalance
- model comparison
- hyperparameter tuning
- model explainability (SHAP)


## Dataset (muhammadfaizan65/hospital-infections-dataset from kagglehub) includes hospital-level data:

- location (state, county)
- infection measure types
- performance scores
- comparison to national benchmarks

Target variable:

compared_to_national - transformed into binary classification:
1 for better than national benchmark, 0 for same or worse.


Originally a multiclass problem (better/same/worse/not available), it was simplified to binary classification due to strong class imbalance,
ambiguous "not available" category and improved model stability.

## Features Used
After experimentation, two setups were explored:

### Minimal feature set:
- state
- measure_name
- score

### Extended feature set (final approach):
- state
- measure_name
- measure_id
- county_name
- score



## Models Tested
- Logistic Regression
- Random Forest
- XGBoost
- Deep Learning: fully connected Neural Network (Keras)
- TabNet (PyTorch)

## Experiments Conducted
- Feature reduction vs expansion
- Random vs group-aware splits
- Hyperparameter tuning (manual grid)
- Neural network architecture search:
-  - layers, neurons, dropout, learning rate
- - batch normalization
- - L1/L2 regularization
- Tree-based model optimization
- TabNet comparison


## Results

XGBoost achieved the best performance:

F1 ≈ 0.54
ROC-AUC ≈ 0.86

## Model Explainability (SHAP)

SHAP analysis revealed:

- score is the most influential feature
- certain measure_name and county_name values also contribute
- model tends to overpredict the "better" class