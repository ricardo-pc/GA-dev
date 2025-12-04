from .simdata1 import X_sim1, y_sim1, R2_sim1
from .simdata2 import X_sim2, y_sim2, true_predictors_sim2, R2_sim2
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

