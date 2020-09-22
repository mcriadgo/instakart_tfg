import preprocess
import cross_validation
from scipy.sparse import csr_matrix


init_df = preprocess.create_df()
filtered_df = preprocess.process_df(init_df, num_users=5000, num_items=8000)
train, test, val = preprocess.train_test_val_split(filtered_df, 0.2, 0.1)
alpha_val = 15
product_user_altered, train_table, mask_train, sparse_train, sparse_val, sparse_test = preprocess.sample_df(train, test, val)


min_iter, min_reg, min_latent_factors, final_error = cross_validation.validate_params(sparse_train, train_table, mask_train)

