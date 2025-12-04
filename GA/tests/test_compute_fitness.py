import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from GA.GA import _compute_fitness

# Test output shapes are correct
def test_output_shape():
    P, p = 10, 5
    gen = np.random.randint(0,2, size = (P,p))
    X = np.random.rand(50,p)
    y = np.random.rand(50)
    penalty = 0.01
    model_type = "linear"
    model_params = None
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits = 5).split(X)

    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, penalty, model_type, 
                                                 model_params, SST, folds)

    assert fitness_raw.shape == (P, )
    assert fitness_pen.shape == (P, )


# Test function returns two arrays
def test_returns_two_arrays():
    gen = np.random.randint(0, 2, size=(10, 5))
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    penalty = 0.01
    model_type = "linear"
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits=5).split(X)
    
    result = _compute_fitness(gen, X, y, penalty, model_type, None, SST, folds)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)

# Test fitness values are reasonable (finite - no NaN or inf)
def test_fitness_values_finite():
    gen = np.random.randint(0, 2, size=(10, 5))
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    penalty = 0.01
    model_type = "linear"
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits=5).split(X)
    
    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, penalty, 
                                                model_type, None, SST, folds)
    
    assert np.all(np.isfinite(fitness_raw))
    assert np.all(np.isfinite(fitness_pen))

# Test function with different model types
def test_tree_model():
    gen = np.random.randint(0,2, size = (5,5))
    gen[0] = np.ones(5, dtype = int)
    X = np.random.rand(50,5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits = 5).split(X)

    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, None,
                                                "tree", None, SST, folds)

    assert fitness_raw.shape == (5,)
    assert fitness_pen.shape == (5,)
    assert np.all(np.isfinite(fitness_raw))

def test_lasso_model():
    gen = np.random.randint(0,2, size = (5,5))
    gen[0] = np.ones(5, dtype = int)
    X = np.random.rand(50,5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits = 5).split(X)

    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, None,
                                                "lasso", None, SST, folds)

    assert fitness_raw.shape == (5,)
    assert fitness_pen.shape == (5,)
    assert np.all(np.isfinite(fitness_raw))


# Test the penalty function works (reduces fitness)
def test_penalty_reduce_fitness():
    gen = np.random.randint(0,2, size=(10,5))
    gen[0] = np.ones(5, dtype = int) # at least one chromosone has predictor selected
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    penalty = 0.1
    model_type = "linear"
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits=5).split(X)
    
    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, penalty, 
                                                model_type, None, SST, folds)
    
    has_predictors = gen.sum(axis=1) > 0
    assert np.all(fitness_pen[has_predictors] <= fitness_raw[has_predictors])

# Test no penalty gives same raw fitness
def test_no_penalty_equal_fitness():
    gen = np.random.randint(0,2, size=(10,5))
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    penalty = None
    model_type = "linear"
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits=5).split(X)
    
    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, penalty, 
                                                model_type, None, SST, folds)
    
    assert np.allclose(fitness_raw, fitness_pen)

# Test chromosones with no predictors get bad fitness
def test_no_predictors_bad_fitness():
    gen = np.zeros((5, 5), dtype=int)  # All chromosomes have no predictors
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    SST = np.sum((y - y.mean())**2)
    folds = KFold(n_splits=5).split(X)
    
    fitness_raw, fitness_pen = _compute_fitness(gen, X, y, None, 
                                                "linear", None, SST, folds)
    
    # All should have very negative fitness
    assert np.all(fitness_raw < -1e8)
    assert np.all(fitness_pen < -1e8)