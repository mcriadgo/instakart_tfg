import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
path_orders = "C:/Users/Marta/Documents/RecommenderSystem/BBDD/orders.csv"
path_prior = "C:/Users/Marta/Documents/RecommenderSystem/BBDD/order_products__train.csv"

def create_df():
    """
        Creates df to insert in ALS algorithm. Only uses train subset
        Merges orders dataframe with train subset dataframe, creates ratings column
        with values from 0 to 1/30
        Drops unused columns and deletes duplicates
        :return not_dup_df: resulting dataframe
    """
    orders = pd.read_csv(path_orders)
    train_orders = pd.read_csv(path_prior)
    merged_df = pd.merge(train_orders, orders, on='order_id', how='left')
    merged_df.dropna(how='any', inplace=True)
    f_df = merged_df[merged_df['days_since_prior_order'] != 0]
    f_df['ratings'] = (f_df['reordered']) / (f_df['days_since_prior_order'])

    f_df.replace([np.inf, 0], np.nan).dropna(how='any', inplace=True)
    f_df = f_df.drop(
        ['eval_set', 'add_to_cart_order', 'reordered', 'order_number', 'order_dow', 'order_hour_of_day',
         'days_since_prior_order'],
        1)
    not_dup_df = f_df.drop_duplicates(['user_id', 'product_id'], keep='first')
    return not_dup_df

def check_sparsity(sparse_matrix):
    """
        Checks sparsity level of matrix
        :param sparse_matrix: matrix to check sparsity
        :return (int) sparsity: sparsity level in per cent
    """
    matrix_size = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    num_purchases = len(sparse_matrix.nonzero()[0])
    sparsity = 100 * (1 - (num_purchases / matrix_size))
    print('The sparsity level of the matrix is %d' % sparsity)
    return sparsity

def process_df(df, num_users, num_items):
    """
        Calls create_df() to obtain a dataframe that can be filtered.
        Dataframe must be filtered with num_users and num_items so that final matrix to nourish ALS algortithm
        is not so sparse
        :param num_users: users to filter
        :param num_items: items to filter
        :return dfFilter (dataframe): filtered dataframe
    """
    # most_freq_products = df['product_id'].value_counts()[:num_items].index.tolist()
    # most_freq_users = df['user_id'].value_counts()[:num_users].index.tolist()
    # dfFilter = df.iloc[most_freq_products,:]
    #user_selected = df.iloc[most_freq_users,:]
    rep_products = pd.DataFrame({'count' : df.groupby("product_id").size()}).reset_index().sort_values("count", ascending=False)
    prod_selected = rep_products["product_id"][:num_items]
    dfFilter_tmp = df[:][df.product_id.isin(prod_selected)]
    rep_user = pd.DataFrame({'count' : dfFilter_tmp.groupby("user_id").size()}).reset_index().sort_values("count", ascending=False)
    users = rep_user["user_id"][:num_users]
    dfFilter = dfFilter_tmp[:][dfFilter_tmp.user_id.isin(users)]
    dfFilter.drop('order_id', 1, inplace=True)
    return dfFilter

def train_test_val_split(data, test_ratio, val_ratio):
    """
        Splits train, test and validation sets
        :param data: original dataframe to split
        :param test_ratio: testing ratio (usually set to 20%)
        :param val_ratio: validation ratio (usually set to 10%)
        :return train_df, test_df, val_df (dataframes): train, test and validation dataframes
    """
    seed = np.random.seed(42)
    shuffle_idx = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)
    train_indices = shuffle_idx[:len(data) - val_size - test_size - 1]
    val_indices = shuffle_idx[(len(data)- val_size - test_size):(len(data)-test_size-1)]
    test_indices = shuffle_idx[(len(data)-test_size):]

    train_df = data.iloc[train_indices]
    test_df = data.iloc[test_indices]
    val_df = data.iloc[val_indices]
    return train_df, test_df, val_df


def sample_df(train_df, test_df, val_df):
    results = {}
    train_table= train_df.pivot(index='user_id', columns='product_id', values='ratings').fillna(0).values

    val_table = val_df.pivot(index='user_id', columns='product_id',
                                                             values='ratings').fillna(0)
    test_table = test_df.pivot(index='user_id', columns='product_id', values='ratings').fillna(0)

    # Every interaction between item and product has to be one, the others zero
    test_table[test_table != 0] = 1
    val_table[val_table != 0] = 1
    train_df_boolean = train_table.copy()
    train_df_boolean[train_df_boolean != 0] = 1
    

    yes_no_array = (val_df != 0).any(axis=1).values
    product_user_altered = np.where(yes_no_array is True)[0].tolist()
    # train_table = train_table
    # sparse_train = csr_matrix(train_table)
    # sparse_val = csr_matrix(val_table)
    # sparse_test = csr_matrix(test_table)

    return product_user_altered, train_table, train_df_boolean


