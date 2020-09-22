# Instakart

This repository contains the development of a recommender system with collaborative filtering and implicit dataset.
It is composed by:
1. __Preprocessing__ : script that creates initial dataframe as a result of merging the two initial files
2. __Cross Validation__: script that runs cross validation in the validation set for hyperparameter tuning
3. __ALS__: script that runs Alternating Least Sqaures algorithm developed manually
4. __Model Performance__: script that checks model performance to see if the final recommendations are optimal

#### Dataset

The dataset has been obtained from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)
but only two files are used:  
- `order_products__train.csv` that contain data related to the training set
- `orders.csv` that contain more data related to each order 

The computed dataset is formed by users identifier, product identifier and rating from each user to each product

#### Algorithm
_Alternating Least Squares_ is the chosen algorithm to develop this recommender system because data is implicit,
it is based on ratings given by users to products. 

This algorithm has been developed manually using [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

