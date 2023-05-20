import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import ndcg_score

def recall_at_k(truth, predicted, k):
    # Flatten the input arrays
    truth = np.ravel(truth)
    predicted = np.ravel(predicted)
    # Sort the predicted values in descending order and get the top-k indices
    top_k_indices = np.argsort(predicted)[::-1][:k]
    # Get the corresponding predicted values for the top-k indices
    top_k_predicted = predicted[top_k_indices]
    # Find the indices where the ground truth is non-zero (positive samples)
    positive_indices = np.nonzero(truth)[0]
    # Find the intersection between the top-k indices and positive indices
    intersection = np.intersect1d(top_k_indices, positive_indices)
    # Calculate recall@k
    if len(positive_indices)==0:
        recall = 0
    else:
        recall = len(intersection) / len(positive_indices)
    return recall

file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Yelp/business_category.dat"
business_category = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Yelp/business_city.dat"
business_city = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Yelp/user_business.dat"
user_business = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Yelp/user_compliment.dat"
user_compliment = pd.read_table(file_path, header=None)
file_path = "C:/Users/ME/PycharmProjects/pythonProject1/venv/Scripts/Yelp/user_user.dat"
user_user = pd.read_table(file_path, header=None)
user_business.columns=['user','item','rating']
business_category.columns = ['item', 'category','1']
business_city.columns = ['item', 'city','1']
user_compliment.columns = ['user', 'compliment','1']

import random
sample_size = 1000000  # Adjust the sample size as needed
max_i = 16239
max_j = 14284
sample = set()
while len(sample) < sample_size:
    i = random.randint(1, max_i)
    j = random.randint(1, max_j)
    sample.add((i, j))

# Generate negative edges by subtracting existing edges from all edges
negative_edges = pd.DataFrame(list(sample - set(user_business.iloc[:,0:2])),columns=["user","item"])
negative_edges["rating"]=0
user_business["rating"]=1
user_business=pd.concat([user_business,negative_edges])

data = pd.merge(user_business,user_compliment.iloc[:,0:2],on = 'user')
data = pd.merge(data,business_city.iloc[:,0:2], on="item")
sparse_features = ["item", "user", "compliment", "city"]
target = ['rating']

# preprocess the sequence feature
category_list = []
for i in range(1, 14284+1):
    category_list.append(list(business_category.loc[business_category['item'] == i, 'category']))
category_length = np.array(list(map(len, category_list)))
max_len = max(category_length)
category_list = pad_sequences(category_list, maxlen=max_len, padding='post', )
category_list = pd.DataFrame(np.column_stack((np.arange(1, 14284+1), category_list)))
category_list.columns = ["item", "A", "B", "C", "D", "E", "F", "G", "I", "J", "K"]
data = pd.merge(data,category_list, on="item")

# 2.count #unique features for each sparse field and generate feature config for sequence feature
fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4)
                          for feat in sparse_features]

use_weighted_sequence = False
if use_weighted_sequence:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('category', vocabulary_size=511 + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                               weight_name='category_weight')]  # Notice : value 0 is for padding for sequence input feature
else:
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('category', vocabulary_size=511 + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                               weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=123)

train_input = {name: train[name] for name in feature_names[0:-1]}
train_input["category"] = np.array(train.iloc[:,5:])
train_input["category_weight"] = np.random.randn(train.shape[0], max_len, 1)
test_input = {name: test[name] for name in feature_names[0:-1]}
test_input["category"] = np.array(test.iloc[:,5:])
test_input["category_weight"] = np.random.randn(test.shape[0], max_len, 1)

es = EarlyStopping(monitor='val_loss', patience=5)
# 4.Define Model,compile and train
from deepctr.models import DeepFM
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import FNN
model = FNN(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import PNN
model = PNN(dnn_feature_columns, task='binary',use_outter=True,kernel_type='mat')
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import CCPM
model = CCPM(linear_feature_columns, dnn_feature_columns, task='binary',conv_filters=(4, 4),dnn_hidden_units=(128,32))
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import WDL
model = WDL(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=(128,64,32)*2)
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import DCN
model = DCN(linear_feature_columns, dnn_feature_columns, task='binary', dnn_hidden_units=(128,64,32)*2)
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import NFM
model = NFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_hidden_units=(128,64,32))
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import xDeepFM
model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_hidden_units=(256,128,64), cin_layer_size=(16, 16))
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import AFM
model = AFM(linear_feature_columns, dnn_feature_columns, task='binary', attention_factor=8)
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

from deepctr.models import AutoInt
model = AutoInt(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)

model = FMAutoint(linear_feature_columns, dnn_feature_columns, task='binary', att_layer_num=2, att_embedding_size=4, att_head_num=1)
model.compile("adam", "binary_crossentropy",  metrics=["binary_crossentropy"] )
history = model.fit(train_input, train["rating"].values,epochs=50, validation_split=0.125, callbacks=[es], batch_size=256)
pred_ans = model.predict(test_input)
rec,ndcg,count=0,0,0
for i in range(1,16240):
    if sum(test_input["user"]==i)>0:
        rec+=recall_at_k(test["rating"][test_input["user"] == i].values.reshape(1, -1),pred_ans[test_input["user"] == i].T, k=10)
        ndcg+=ndcg_score(test["rating"][test_input["user"]==i].values.reshape(1,-1),pred_ans[test_input["user"]==i].T, k=10)
        count+=1
rec=rec/count
ndcg=ndcg/count
print(rec,ndcg)