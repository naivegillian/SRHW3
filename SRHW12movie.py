import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/movie_genre.dat"
movie_genre = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/movie_movie(knn).dat"
movie_movie_knn = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/user_age.dat"
user_age = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/user_movie.dat"
user_movie = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/user_occupation.dat"
user_occupation = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Movielens/user_user(knn).dat"
user_user_knn = pd.read_table(file_path, header=None)
user_movie.columns = ['user', 'item', 'rating', 'time']
user_age.columns = ['user', 'age']
user_occupation.columns = ['user', 'occupation']
movie_genre.columns = ['item', 'genre']
data = pd.merge(user_movie,user_age,on = 'user')
data = pd.merge(data,user_occupation,on = 'user')
sparse_features = ["item", "user", "age", "occupation"]
target = ['rating']

# preprocess the sequence feature
genres_list = []
for i in range(1, 1682+1):
    genres_list.append(list(movie_genre.loc[movie_genre['item'] == i, 'genre']))
genres_length = np.array(list(map(len, genres_list)))
max_len = max(genres_length)
genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
genres_list = pd.DataFrame(np.column_stack((np.arange(1, 1683), genres_list)))
genres_list.columns = ["item", "A", "B", "C", "D", "E", "F"]
data = pd.merge(data,genres_list, on="item")

# 2.count #unique features for each sparse field and generate feature config for sequence feature
fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                          for feat in sparse_features]

use_weighted_sequence = False
if use_weighted_sequence:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genre', vocabulary_size=18 + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                               weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
else:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genre', vocabulary_size=18 + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                               weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=123)

train_input = {name: train[name] for name in feature_names[0:-1]}
train_input["genre"] = np.array(train.iloc[:,-6:])
train_input["genre_weight"] = np.random.randn(train.shape[0], max_len, 1)
test_input = {name: test[name] for name in feature_names[0:-1]}
test_input["genre"] = np.array(test.iloc[:,-6:])
test_input["genre_weight"] = np.random.randn(test.shape[0], max_len, 1)

# 4.Define Model,compile and train
from deepctr.models import DeepFM
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
es = EarlyStopping(monitor='val_mse', patience=3)
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import FNN
model = FNN(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import PNN
model = PNN(dnn_feature_columns, task="regression",use_inner=True,kernel_type='vec')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import CCPM
model = CCPM(linear_feature_columns, dnn_feature_columns, task='regression',conv_filters=(4, 4),dnn_hidden_units=(128,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import WDL
model = WDL(linear_feature_columns, dnn_feature_columns, task='regression',dnn_hidden_units=(128,64,32)*2)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import DCN
model = DCN(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(128,64,32)*2)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import NFM
model = NFM(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(128,64,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import xDeepFM
model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', cin_layer_size=(16, 16))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import AFM
model = AFM(linear_feature_columns, dnn_feature_columns, task='regression', attention_factor=8)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import AutoInt
model = AutoInt(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

model = FMAutoint(linear_feature_columns, dnn_feature_columns, task='regression', att_layer_num=2, att_embedding_size=4, att_head_num=1)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))