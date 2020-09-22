import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BBDD_DIR = os.path.join(ROOT_DIR, 'BBDD')
ORDERS_PATH = os.path.join(BBDD_DIR, 'orders.csv')
TRAIN_ORDERS_PATH = os.path.join(BBDD_DIR, 'order_products__train.csv')


# Cross Validation
PARAMS_GRID = {'iterations':[50, 100, 130],
              'reg': [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.5],
              'latent_factors':[50, 75, 100, 130, 150]}

# Model performance
ITERATIONS = [50, 75, 100, 130, 150, 165]
REGULARIZATION_PARAM = [0.001, 0.005, 0.01, 0.05, 0.06, 0.1, 0.5]