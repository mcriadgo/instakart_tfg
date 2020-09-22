import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import implicit

from src import config


def dcg_at_k(r, k, method=0):
    """
        Computes Discounted Cumulative Gain at k
        :param r: list of relevance of each elements
        :param k: top k to compute DCG
        :param method:
        :return (double) dcg: discounted cumulative gain result
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def mAP_nDCG_k(als_user_item, trainSparse, altered_users, testSparse, k, ran=False):
    """

    :param als_user_item: matrix reconstructed from ALS algorithm
    :param trainSparse: original sparse training matrix
    :param altered_users: users that have rated products
    :param testSparse: validation or test sparse matrix to test results with
    :param k:
    :param ran: boolean parameter to set random elements as y_score or elements from original matrix
    :return:
    """
    sum_AP = 0
    sum_nDCG = 0
    for user in altered_users:
        if ran is False:
            y_score = als_user_item[user, :].tolist()
        else:
            y_score = [random.random() for _ in range(trainSparse.shape[1])]
        y_true = testSparse.toarray()[user, :]
        y_true = y_true[:trainSparse.shape[1]]
        df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
        df_orderedPred = df.sort_values(by="y_score", ascending=False)

        y_true_k = df_orderedPred["y_true"].values[:k]
        y_score_k = df_orderedPred["y_score"].values[:k]
        av_prec = average_precision_score(y_true_k, y_score_k)
        if str(av_prec) == "nan":
            av_prec = 0
        sum_AP += av_prec

        dcg = dcg_at_k(y_true_k, k)
        dcg_max = dcg_at_k(y_score_k, k)  # todos los elementos de la lista son 1
        sum_nDCG += dcg / dcg_max

    mAP = sum_AP / len(altered_users)
    nDCG = sum_nDCG / len(altered_users)

    return mAP, nDCG


def precision_recall_F1_at_k(sparse_train, altered_users, predictions_list, test_set, k, ran):
    """

    :param sparse_train: sparse original training matrix
    :param altered_users: users that have rated products
    :param predictions_list: [users result sparse matrix, items result sparse matrix]
    :param test_set: test or validation sparse matrix
    :param k:
    :param ran: boolean flag to set if random performance is better
    :return:
    """
    precisions = []
    recalls = []
    f1 = []
    for user in range(sparse_train.shape[0]):
        rankingTestItems = getRankingPos_test(user, sparse_train, predictions_list, test_set, ran)
        size_HitSet = sum(rankingTestItems <= k)  # sum of first k elements
        size_testSet = len(rankingTestItems)
        recall = size_HitSet / size_testSet # recall = true_positive / (true_positive+false_negative)
        precision = size_HitSet / k  # precision = true_positive / (true_positive+false_positive)
        F1 = 2 * precision * recall / (recall + precision)
        if str(F1) == "nan":
            F1 = 0
        recalls.append(recall)
        precisions.append(precision)
        f1.append(F1)
    return np.mean(np.asarray(precisions)), np.mean(np.asarray(recalls)), np.mean(np.asarray(f1))

def getRankingPos_test(user, training_set, predictions_list, test_set, ran):
    """
        Creates pandas Series that correspond to the most ranked products in results of ALS algorithm
        :param user:
        :param training_set: sparse original training matrix
        :param predictions_list: [users result sparse matrix, items result sparse matrix]
        :param test_set: test or validation sparse matrix
        :param ran: boolean flag to set if random performance is better
        :return:
    """
    product_vecs = predictions_list[1]
    training_row = training_set[user, :].toarray().reshape(-1)

    zero_inds = np.where(training_row == 0)
    if ran is False:
        user_vec = predictions_list[0][user, :]
        predictions = user_vec.dot(product_vecs).toarray()[0, zero_inds].reshape(-1)
    else:
        predictions = [random.random() for _ in range(training_set.shape[1])]

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values(by=0, ascending=False)
    predictions_df.columns = ["Predictions"]
    predictions_df["num_ranking"] = range(1, predictions_df.shape[0] + 1)

    test = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
    test_df = pd.DataFrame(test)
    rankedProducts_test = test_df[test != 0]
    ranking_testProducts = predictions_df.loc[rankedProducts_test.index, "num_ranking"]
    return ranking_testProducts

def f1_score_set_iterations(altered, sparse_train, sparse_val):
    """

    :param sparse_train:
    :param sparse_val:
    :return:
    """
    map_array = []
    ndcg_array = []
    for k in config.ITERATIONS:
        model = implicit.als.AlternatingLeastSquares(factors=k, regularization=0.05, iterations=25,
                                                     calculate_training_loss=True)
        model.fit(sparse_train)
        users_factors = model.item_factors
        item_factors = model.user_factors
        result_factors = np.dot(users_factors, item_factors)
        mAP, nDCG = mAP_nDCG_k(result_factors, sparse_train, altered, sparse_val, k=3)
        map_array.append(mAP)
        ndcg_array.append(nDCG)

    plt.plot(config.ITERATIONS, f1_array, '.-')
    plt.grid(True)
    plt.title('mAP vs. Factores latentes (25 iteraciones)')
    plt.xlabel('Número de factores latentes')
    plt.ylabel('Valor de mAP')

def map_lambda_reg(altered, sparse_train, sparse_test, sparse_val):
    """

    :param sparse_train:
    :param sparse_val:
    :return:
    """

    map_array = []
    ndcg_array = []
    for k in config.REGULARIZATION_PARAM:
        model = implicit.als.AlternatingLeastSquares(factors=120, regularization=k, iterations=25,
                                                     calculate_training_loss=True)
        model.fit(sparse_train)
        users_factors = model.item_factors
        item_factors = model.user_factors
        result_factors = np.dot(users_factors, item_factors.T)
        mAP, nDCG = mAP_nDCG_k(result_factors, sparse_train, altered, sparse_test)
        map_array.append(mAP)
        ndcg_array.append(nDCG)

    plt.plot(config.REGULARIZATION_PARAM, map_array, '.-')
    plt.grid(True)
    plt.xlabel('Valor del hiperparámetro de regularización')
    plt.ylabel('Valor de mAP')