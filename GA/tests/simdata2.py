import numpy as np

# ====================================================
# Simulate data with non-linear relationships
# to test GA with decision tree model
# ====================================================
np.random.seed(123)
n = 500
p = 10

# Generate predictors
X_sim2 = np.random.normal(size=(n, p))

# Create non-linear relationships with predictors 1, 4, and 6
# Predictor 1: quadratic effect
# Predictor 4: interaction-like (combines with itself non-linearly)
# Predictor 6: threshold effect
y_sim2 = (
    2.0 * X_sim2[:, 1]**2 +           # Quadratic
    1.5 * np.abs(X_sim2[:, 4]) +      # Absolute value (non-linear)
    1.0 * (X_sim2[:, 6] > 0) +        # Threshold/step function
    np.random.normal(scale=0.5, size=n)
)

# noise
sigma = 0.5

true_predictors_sim2 = [1, 4, 6]

# Approximate R^2
signal_var = np.var(y_sim2 - np.random.normal(scale=0.5, size=n))
R2_sim2 = signal_var / (signal_var + sigma**2)