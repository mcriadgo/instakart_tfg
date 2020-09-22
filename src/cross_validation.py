import numpy as np
import implicit

from src import config


def get_error(A, resultMatrix, R):
    return np.sum((R * (A - resultMatrix)) ** 2) / np.sum(R)


def iter_cross_val(sparse_train, item_user_matrix, mask_item_user):
    """
        Cross validation method
        :param sparse_item_user_matrix: sparse matrix obtained from preprocessing
        :param item_user_matrix: matrix obtained from preprocessing for training data
        :param item_user_matrix_bool: mask of matrix obtained from preprocessing for training data
        :param (int) min_iter: number of iterations to obtain min error
    """
    min_error = 9999
    min_iter = 0

    for i in config.PARAMS_GRID.get('iterations'):
        print('Process for iteration %s' % str(i))
        imp_als = implicit.als.AlternatingLeastSquares(factors=120, regularization=0.01, iterations=i,
                                                       calculate_training_loss=True)
        imp_als.fit(sparse_train)
        users_factors = imp_als.item_factors
        item_factors = imp_als.user_factors
        error_iter = get_error(item_user_matrix, np.dot(users_factors, item_factors.T), mask_item_user)
        if error_iter < min_error:
            min_error = error_iter
            min_iter = i
    return min_iter


def reg_cross_val(sparse_train, item_user_matrix, mask_item_user, min_iter):
    """
        Cross validation method
        :param sparse_train: sparse matrix obtained from preprocessing
        :param item_user_matrix: matrix obtained from preprocessing for training data
        :param mask_item_user: mask of matrix obtained from preprocessing for training data
        :return (int) min_reg: reguralization factor to obtain min error
    """
    min_error = 9999
    min_reg = 0

    for reg in config.PARAMS_GRID.get('reg'):
        print('Regularization parameter %s' % str(reg))
        imp_als = implicit.als.AlternatingLeastSquares(factors=100, regularization=reg, iterations=min_iter,
                                                       calculate_training_loss=True)
        imp_als.fit(sparse_train)
        users_factors = imp_als.item_factors
        item_factors = imp_als.user_factors

        error_reg = get_error(item_user_matrix, np.dot(users_factors, item_factors.T), mask_item_user)
        if error_reg < min_error:
            min_error = error_reg
            min_reg = reg
    return min_reg


def lf_cross_val(sparse_train, item_user_matrix, mask_item_user, min_iter, min_reg):
    """
        Cross validation method for latent factors
        :param sparse_train: sparse matrix obtained from preprocessing
        :param item_user_matrix: matrix obtained from preprocessing for training data
        :param mask_item_user: mask of matrix obtained from preprocessing for training data
        :return (int) min_latentfactors: number of latent factors to obtain min error
        :return (int) min_error: minimum error
    """
    min_latentfactors = 0
    min_error = 9999

    for lf in config.PARAMS_GRID.get('latent_factors'):
        print('Latent factors %s' % str(lf))
        imp_als = implicit.als.AlternatingLeastSquares(factors=lf, regularization=min_reg, iterations=min_iter,
                                                       calculate_training_loss=True)
        imp_als.fit(sparse_train)
        users_factors = imp_als.item_factors
        item_factors = imp_als.user_factors
        error_reg = get_error(item_user_matrix, np.dot(users_factors, item_factors.T), mask_item_user)
        if error_reg < min_error:
            min_error = error_reg
            min_latentfactors = lf
    return min_latentfactors, min_error


def validate_params(sparse_train, item_user_matrix, mask_item_user):

    min_iter = iter_cross_val(sparse_train, item_user_matrix, mask_item_user)
    min_reg = reg_cross_val(sparse_train, item_user_matrix, mask_item_user, min_iter)
    min_latent_factors, final_error = lf_cross_val(sparse_train, item_user_matrix, mask_item_user,
                                                   min_iter, min_reg)
    return min_iter, min_reg, min_latent_factors