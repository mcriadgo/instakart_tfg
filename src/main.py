import preprocess
import cross_validation
from scipy.sparse import csr_matrix


init_df = preprocess.create_df()
filtered_df = preprocess.process_df(init_df, num_users=500, num_items=2500)
train, test, val = preprocess.train_test_val_split(filtered_df, 0.2, 0.1)
alpha_val = 15
product_user_altered, train_table, train_df_boolean = preprocess.sample_df(train, test, val)
train_table= train.pivot(index='user_id', columns='product_id', values='ratings').fillna(0).values

# sparse_item_user = csr_matrix((train['ratings'], (train['product_id'], train['user_id'])))
sparse_item_user = csr_matrix(train_table)

sparse_user_item = csr_matrix((train['ratings'], (train['user_id'], train['product_id'])))
errors = cross_validation.validate_params(sparse_item_user, train_table, train_df_boolean)
