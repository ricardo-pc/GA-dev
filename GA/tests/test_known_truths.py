from .simdata1 import X_sim1, y_sim1, R2_sim1
import GA

def test_output():
    result_sim1 = GA.select(X_sim1, y_sim1, penalty=0.01)
    assert result_sim1["selected"] == [0, 3, 7]

    assert abs(result_sim1["R2"] - R2_sim1) < 0.05
