import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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
book_author.columns = ['item', 'author']
book_publisher.columns = ['item', 'publisher']
book_year.columns = ['item', 'year']
user_group.columns = ['user', 'group']
user_location.columns = ['user', 'location']

data = pd.merge(user_book,user_group,on = 'user')
data = pd.merge(data,user_location,on = 'user')
data = pd.merge(data,book_author, on="item")
data = pd.merge(data,book_publisher, on="item")
data = pd.merge(data,book_year, on="item")
sparse_features = ["item", "user", "group", "location", "author", "publisher", "year"]
target = ['rating']

# 2.count #unique features for each sparse field and generate feature config for sequence feature
fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                          for feat in sparse_features]

linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=123)

train_input = {name: train[name] for name in feature_names}
test_input = {name: test[name] for name in feature_names}

# 4.Define Model,compile and train
from deepctr.models import DeepFM
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
es = EarlyStopping(monitor='val_mse')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import FNN
model = FNN(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import PNN
model = PNN(dnn_feature_columns, task="regression",use_inner=True,kernel_type='mat')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*16,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import CCPM
model = CCPM(linear_feature_columns, dnn_feature_columns, task='regression',conv_filters=(4, 4),dnn_hidden_units=(128,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import WDL
model = WDL(linear_feature_columns, dnn_feature_columns, task='regression',dnn_hidden_units=(128,64,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import DCN
model = DCN(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(128,64,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import NFM
model = NFM(linear_feature_columns, dnn_feature_columns, task='regression', dnn_hidden_units=(128,64,32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import xDeepFM
model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', cin_layer_size=(32, 32))
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import AFM
model = AFM(linear_feature_columns, dnn_feature_columns, task='regression', attention_factor=8)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

from deepctr.models import AutoInt
model = AutoInt(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4,epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))

model = FMAutoint(linear_feature_columns, dnn_feature_columns, task='regression', att_layer_num=2, att_embedding_size=4, att_head_num=1)
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_input, train["rating"].values,batch_size=256*4, epochs=5, validation_split=0.125, callbacks=[es])
pred_ans = model.predict(test_input)
print(np.sqrt(mean_squared_error(test["rating"].values, pred_ans)))