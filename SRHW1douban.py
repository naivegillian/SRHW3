import pandas as pd
import numpy as np
import surprise
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise import accuracy


# Set the file path
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/book_author.dat"
book_author = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/book_publisher.dat"
book_publisher = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/book_year.dat"
book_year = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/user_book.dat"
user_book = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/user_group.dat"
user_group = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/user_location.dat"
user_location = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Douban Book/user_user.dat"
user_user = pd.read_table(file_path, header=None)
user_book.columns=['user','item','rating']

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_book[["user", "item", "rating"]], reader)

sim_options = {
    "name": "cosine",
    "user_based": False,
}
algo = KNNWithMeans(sim_options=sim_options)

trainset, testset = train_test_split(data, test_size=0.2,random_state=123)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

import random
all_values = list(range(len(user_book)))
random.seed(123)
random.shuffle(all_values)
all_values=np.array(all_values)
trainset = Dataset.load_from_df(user_book.loc[all_values[:round(len(user_book)*0.8)]][["user", "item", "rating"]], reader)
testset = Dataset.load_from_df(user_book.loc[all_values[round(len(user_book)*0.8):]][["user", "item", "rating"]], reader)

sim_options = {
    "name": ["pearson"],
    "min_support": [3],
    "k": [3],
    "user_based": [False],
}
param_grid = {"sim_options": sim_options}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=8, refit=True)
gs.fit(trainset)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

algo = KNNWithMeans(sim_options=gs.best_params["rmse"]["sim_options"])
algo.fit(trainset.build_full_trainset())
predictions = algo.test(testset.build_full_trainset().build_testset())
accuracy.rmse(predictions)
algo.predict(1, 1)
data.df[(data.df['user'] == 1) & (data.df['item'] == 1)]

import spotlight
from spotlight.cross_validation import random_train_test_split
from spotlight.interactions import Interactions
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from surprise.model_selection import train_test_split

dataset = Interactions(user_ids=user_book['user'].values,
                            item_ids=user_book['item'].values,
                            ratings=user_book['rating'].values)

train, test = random_train_test_split(dataset,test_percentage=0.2)

model = ExplicitFactorizationModel(n_iter=10, embedding_dim=2)
model.fit(train)

rmse = rmse_score(model, test)
print(rmse)
model.predict(1).shape


from sklearn.model_selection import KFold
kf = KFold(n_splits=8, shuffle=True, random_state=123)
datasetarray=np.column_stack((train.user_ids,train.item_ids,train.ratings))

# Define the range of embedding dimensions to try
embedding_dims = [2, 8, 32]

# Iterate over the different embedding dimensions
for embedding_dim in embedding_dims:
    rmse_scores = []
    for X_train,X_test in kf.split(datasetarray):
        trainset = Interactions(user_ids=datasetarray[X_train][:,0],
                               item_ids=datasetarray[X_train][:,1],
                               ratings=datasetarray[X_train][:,2])
        testset = Interactions(user_ids=datasetarray[X_test][:,0],
                               item_ids=datasetarray[X_test][:,1],
                               ratings=datasetarray[X_test][:,2])

        model = ExplicitFactorizationModel(n_iter=10, embedding_dim=embedding_dim)
        model.fit(trainset)

        rmse = rmse_score(model, testset)
        rmse_scores.append(rmse)
        print(rmse)

    # Compute the mean RMSE across the different folds
    mean_rmse = sum(rmse_scores) / len(rmse_scores)

    print("Embedding dim:", embedding_dim)
    print("Mean RMSE:", mean_rmse)


from spotlight.evaluation import precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

model = ImplicitFactorizationModel(n_iter=10, embedding_dim=8, loss='bpr')
model.fit(train)

recall = precision_recall_score(model, test, k=10)[1]
print(np.mean(recall))

for embedding_dim in embedding_dims:
    recall_scores = []
    for X_train, X_test in kf.split(datasetarray):
        trainset = Interactions(user_ids=datasetarray[X_train][:, 0],
                                item_ids=datasetarray[X_train][:, 1],
                                ratings=datasetarray[X_train][:, 2])
        testset = Interactions(user_ids=datasetarray[X_test][:, 0],
                               item_ids=datasetarray[X_test][:, 1],
                               ratings=datasetarray[X_test][:, 2])

        model = ImplicitFactorizationModel(n_iter=10, embedding_dim=embedding_dim,loss='bpr')
        model.fit(trainset)

        recall = np.mean(precision_recall_score(model, testset, k=10)[1])
        recall_scores.append(recall)
        print(recall)

    # Compute the mean RMSE across the different folds
    mean_recall = sum(recall_scores) / len(recall_scores)

    print("Embedding dim:", embedding_dim)
    print("Mean recall:", mean_recall)

from sklearn.metrics import ndcg_score
avg=0
for i in range(1,13025):
    predictions=model.predict(i)
    ifyes = np.zeros(22348)
    ifyes[test.item_ids[test.user_ids == i]] = 1
    avg+=ndcg_score(np.asarray([ifyes]),np.asarray([predictions]), k=10)
avg=avg/13025
print(avg)