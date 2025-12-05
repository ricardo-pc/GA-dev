from .simdata1 import X_sim1, y_sim1, R2_sim1
from .simdata2 import X_sim2, y_sim2, true_predictors_sim2, R2_sim2
from .simdata3 import X_sim3, y_sim3, important_groups_sim3, R2_sim3
import GA

def test_output():
    result_sim1 = GA.select(X_sim1, y_sim1, penalty=0.01)
    assert result_sim1["selected"] == [0, 3, 7]

    assert abs(result_sim1["R2"] - R2_sim1) < 0.05

def test_nonlinear_tree():
    result_sim2 = GA.select(X_sim2, y_sim2, model_type="tree", penalty=0.01)
    
    # Check that at least 2 out of 3 true predictors are selected
    selected = result_sim2["selected"]
    matches = sum(pred in selected for pred in true_predictors_sim2)
    assert matches >= 2, f"Expected at least 2 of {true_predictors_sim2}, got {selected}"
    
    # Check reasonable number of predictors (not too many extras)
    assert len(selected) <= 4, f"Selected too many predictors: {len(selected)}"
    
    # Check R^2 is reasonable
    assert result_sim2["R2"] > 0.5, f"R^2 too low: {result_sim2['R2']}"

def test_correlated_lasso():
    result_sim3 = GA.select(X_sim3, y_sim3, model_type="lasso", penalty=0.01)
    
    # Check that at least one predictor from each important group is selected
    groups_found = 0
    for group in important_groups_sim3:
        if any(pred in result_sim3["selected"] for pred in group):
            groups_found += 1

    assert groups_found >= 2, f"Expected predictors from at least 2 groups, found {groups_found}"

    # Check reasonable number of predictors
    assert 2 <= len(result_sim3["selected"]) <= 4, f"Expected 2-4 predictors, got {len(selected)}"
    
    # Check R^2 is reasonably close to theoretical value
    assert abs(result_sim3["R2"] - R2_sim3) < 0.01, f"R^2 {result_sim3['R2']} too far from {R2_sim3}"
    

