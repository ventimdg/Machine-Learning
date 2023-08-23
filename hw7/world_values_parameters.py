import numpy as np

regression_knn_parameters = {
    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression
    ##### TODO(4f): Change the weighting ##### 
    # 'knn__weights': ['uniform']
    'knn__weights': ['distance']
}