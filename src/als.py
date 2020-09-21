import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import implicit
import matplotlib.pyplot as plt


def run_manual_als(original_matrix, lambda_, n_factors, max_iter):
    """
        Runs als manually. Initialize two random matrices with original_matrix shape. By Ridge Regression
        fits every row of user matrix and then every column of item matrix.
        Row for users because users are indexes, columns for items because items are columns in original_matrix.
        Computes RMSE for each iteration
        :param original_matrix
        :param lambda_: regularization term for Ridge regression
        :param n_factors: latent factors
        :param max_iter: iterations to converge als algorithm
        :return (ndarray) users: user matrix data obtained from ALS algorithm
        :return (ndarray) items: item matrix data obtained from ALS algorithm
        :return (list) errors: list of errors for every iteration
    """
    (Nu, Ni) = original_matrix.shape
    users = np.random.rand(Nu, n_factors)
    items = np.random.rand(n_factors, Ni)
    errors_per_iter = []
    errors = []
    rmse = 0
    for iteration in range(max_iter):
        for u in range(Nu):
            if original_matrix[u, :].sum() == 0:
                users[u] = 0
            else:
                clf = Ridge(alpha=lambda_, fit_intercept=False, max_iter=50, tol=0.01)
                clf.fit(items.T, original_matrix[u, :].toarray()[0])
                users[u] = clf.coef_

        RMSE_per_iter = 0

        for i in range(Ni):
            if original_matrix[:, i].sum() == 0:
                items[:, i] = 0
            else:
                clf = Ridge(alpha=lambda_, fit_intercept=False, max_iter=50, tol=0.01)
                clf.fit(users, original_matrix[:, i].toarray()[:, 0])
                items[:, i] = clf.coef_
                RMSE_per_iter = RMSE_per_iter + get_error_RMSE(original_matrix[:, i].toarray()[:, 0], users,
                                                               items[:, i])
        rmse = rmse + get_error_RMSE(original_matrix[:, i].toarray()[:, 0], users, items[:, i])

        errors_per_iter.append(RMSE_per_iter)
        errors.append(rmse)
    return users, items, errors


def get_error_RMSE(ratings, users, items):
    """
        Computes RMSE
        :param ratings: original matrix where users are indexes, items are columns
        :param users: user matrix data obtained from ALS algorithm
        :param items: item matrix data obtained from ALS algorithm
        :return: root mean squared error
    """
    product = np.dot(users, items)
    return np.sqrt(mean_squared_error(ratings, product))

def runALS(A, R, n_factors, n_iterations, lambda_):
    """
        Runs Alternating Least Squares algorithm in order to calculate matrix.
        :param A: User-Item Matrix with ratings
        :param R: User-Item Matrix with 1 if there is a rating or 0 if not
        :param n_factors: How many factors each of user and item matrix will consider
        :param n_iterations: How many times to run algorithm
        :param lambda_: Regularization parameter
        :return users, items: user and item matrix resulting from ALS algorithm
    """

    n, m = A.shape
    users = np.random.rand(n, n_factors)
    items = np.random.rand(n_factors, m)

    def get_error(A, users, items, R):
        return np.sum((R * (A - np.dot(users, items))) ** 2) / np.sum(R)

    MSE_List = []
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            users[i] = np.linalg.solve(np.dot(items, np.dot(np.diag(Ri), items.T)) + lambda_ * np.eye(n_factors),
                                       np.dot(items, np.dot(np.diag(Ri), A[i].T))).T

        for j, Rj in enumerate(R.T):
            items[:, j] = np.linalg.solve(np.dot(users.T, np.dot(np.diag(Rj), users)) + lambda_ * np.eye(n_factors),
                                          np.dot(users.T, np.dot(np.diag(Rj), A[:, j])))

        MSE_List.append(get_error(A, users, items, R))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(range(1, len(MSE_List) + 1), MSE_List)
    # plt.ylabel('Error')
    # plt.xlabel('Iteration')
    # plt.title('Python Implementation MSE by Iteration \n with %d users and %d movies' % A.shape)
    # plt.savefig('Python MSE Graph.pdf', format='pdf')
    # plt.show()
    return users, items

