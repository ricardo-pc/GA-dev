import numpy as np

# ====================================================
# Simulate data with correlated predictors
# to test GA with lasso model
# ====================================================
np.random.seed(456)
n = 500
p = 12

# Generate base predictors
X_base = np.random.normal(size=(n, 3))

# Create correlated copies
X_sim3 = np.zeros((n, p))
X_sim3[:, 0] = X_base[:, 0]  # Original predictor 1
X_sim3[:, 1] = X_base[:, 0] + np.random.normal(scale=0.3, size=n)  # Correlated with 0
X_sim3[:, 2] = X_base[:, 0] + np.random.normal(scale=0.3, size=n)  # Correlated with 0

X_sim3[:, 5] = X_base[:, 1]  # Original predictor 2
X_sim3[:, 6] = X_base[:, 1] + np.random.normal(scale=0.3, size=n)  # Correlated with 5

X_sim3[:, 9] = X_base[:, 2]  # Original predictor 3
X_sim3[:, 10] = X_base[:, 2] + np.random.normal(scale=0.3, size=n)  # Correlated with 9

# Fill remaining with noise
for i in [3, 4, 7, 8, 11]:
    X_sim3[:, i] = np.random.normal(size=n)

# True relationship uses predictors from groups 0-2, 5-6, 9-10
beta = np.zeros(p)
beta[0] = 2.0   # Group 1
beta[5] = -1.5  # Group 2
beta[9] = 1.0   # Group 3

# Add some noise
sigma = 0.5

# Generate response
eps = np.random.normal(scale=0.5, size=n)
y_sim3 = X_sim3 @ beta + eps

# Important predictor groups
important_groups_sim3 = [[0, 1, 2], [5, 6], [9, 10]]

# Theoretical R^2
signal_var = np.var(X_sim3 @ beta)
R2_sim3 = signal_var / (signal_var + sigma**2)