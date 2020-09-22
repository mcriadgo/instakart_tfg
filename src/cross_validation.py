import numpy as np

from src import config
from src import als

def get_error(A, users, items, R):
    return np.sum((R * (A - np.dot(users, items))) ** 2) / np.sum(R)

def validate_params(sparse_item_user_matrix, item_user_matrix, item_user_matrix_bool):
    """
        Cross validation method
        :param sparse_item_user_matrix: sparse matrix obtained from preprocessing
        :param item_user_matrix: matrix obtained from preprocessing for training data
        :param item_user_matrix_bool: mask of matrix obtained from preprocessing for training data
        :return (list) errors: list of errors for every possibility of regularization parameters,
        iterations to converge and latent factors.
    """
    errors = []
    for i in config.PARAMS_GRID.get('iterations'):
        print('Process for iteration %s' % str(i))
        for reg in config.PARAMS_GRID.get('reg'):
            print('Regularization parameter %s' % str(reg))
            for lf in config.PARAMS_GRID.get('latent_factors'):
                print('Latent factors %s' % str(lf))
                #imp_als = implicit.als.AlternatingLeastSquares(factors=lf, regularization=reg, iterations=iter, calculate_training_loss=True)
                #imp_als.fit(sparse_item_user_matrix)
                #users_factors = imp_als.item_factors
                #item_factors = imp_als.user_factors
                users, items, errors = als.run_manual_als(sparse_item_user_matrix, reg, lf, i)
                
                errors.append(get_error(item_user_matrix, users, items, item_user_matrix_bool))
                print(errors)
    return errors