import numpy as np
import pandas as pd
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


iterations = [50, 75, 100, 130, 150, 165]
reg = [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.5]

    
def dcg_at_k(r, k, method=0):
    """

    :param r:
    :param k:
    :param method:
    :return:
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


def mAP_nDCG_k(reconstTrainMatrix, trainSparse, altered_users, testSparse, k, ran=False):
    """

    :param reconstTrainMatrix:
    :param trainSparse:
    :param altered_users:
    :param testSparse:
    :param k:
    :param ran:
    :return:
    """
    sum_AP = 0
    sum_nDCG = 0
    for user in altered_users:
        if ran is False:
            y_score = reconstTrainMatrix[user, :].tolist()
        else:
            y_score = [random.random() for test in range(trainSparse.shape[1])]
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


def precision_recall_F1_at_k(training_set, altered_users, predictions_list, test_set, k, ran):
    """

    :param training_set:
    :param altered_users:
    :param predictions_list:
    :param test_set:
    :param k:
    :param ran:
    :return:
    """
    precisions = []
    recalls = []
    f1 = []
    for user in range(training_set.shape[0]):
        rankingTestSongs = getRankingPos_test(user, training_set, predictions_list, test_set, ran)
        size_HitSet = sum(rankingTestSongs <= k)
        size_testSet = len(rankingTestSongs)
        recall = size_HitSet / size_testSet
        precision = size_HitSet / k
        F1 = 2 * precision * recall / (recall + precision)
        if str(F1) == "nan":
            F1 = 0
        recalls.append(recall)
        precisions.append(precision)
        f1.append(F1)
    return np.mean(np.asarray(precisions)), np.mean(np.asarray(recalls)), np.mean(np.asarray(f1))

def getRankingPos_test(user, training_set, predictions_list, test_set, ran):
    """

    :param user:
    :param training_set:
    :param predictions_list:
    :param test_set:
    :param ran:
    :return:
    """
    item_vecs = predictions_list[1]
    training_row = training_set[user, :].toarray().reshape(-1)

    zero_inds = np.where(training_row == 0)
    if ran is False:
        user_vec = predictions_list[0][user, :]
        predictions = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
    else:
        predictions = [random.random() for test in range(training_set.shape[1])]

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values(by=0, ascending=False)
    predictions_df.columns = ["Predictions"]
    predictions_df["num_ranking"] = range(1, predictions_df.shape[0] + 1)

    test = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
    test_df = pd.DataFrame(test)
    heardSongs_test = test_df[test != 0]
    ranking_testSongs = predictions_df.loc[heardSongs_test.index, "num_ranking"]
    return ranking_testSongs

def f1_score_set_iterations(sparse_train, sparse_val):
    """

    :param sparse_train:
    :param sparse_val:
    :return:
    """
    map_array = []
    ndcg_array = []
    for k in iterations:
        model = implicit.als.AlternatingLeastSquares(factors=k, regularization=0.05, iterations=25,
                                                     calculate_training_loss=True)
        model.fit(sparse_train)
        users_factors = model.item_factors
        item_factors = model.user_factors
        result_factors = np.dot(users_factors, item_factors)
        mAP, nDCG = mAP_nDCG_k(result_factors, sparse_train, altered, sparse_val, k=3)
        map_array.append(mAP)
        ndcg_array.append(nDCG)

    plt.plot(.iterations, f1_array, '.-')
    plt.grid(True)
    plt.title('mAP vs. Factores latentes (25 iteraciones)')
    plt.xlabel('Número de factores latentes')
    plt.ylabel('Valor de mAP')

def map_lambda_reg(, sparse_train, sparse_val):
    """

    :param sparse_train:
    :param sparse_val:
    :return:
    """

    map_array = []
    ndcg_array = []
    for k in reg:
        model = implicit.als.AlternatingLeastSquares(factors=120, regularization=k, iterations=25,
                                                     calculate_training_loss=True)
        model.fit(sparse_train)
        users_factors = model.item_factors
        item_factors = model.user_factors
        result_factors = np.dot(users_factors, item_factors.T)
        mAP, nDCG = mAP_nDCG_k(result_factors, sparse_train, altered, sparse_test)
        map_array.append(mAP)
        ndcg_array.append(nDCG)

    plt.plot(.reg, map_array, '.-')
    plt.grid(True)
    plt.xlabel('Valor del hiperparámetro de regularización')
    plt.ylabel('Valor de mAP')