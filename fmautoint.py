# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from itertools import chain

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Concatenate, Dense

from deepctr.feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM, InteractingLayer
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input



def FMAutoint(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,),
            att_layer_num=3, att_embedding_size=8, att_head_num=2, att_res=True,
            dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, seed=1024,
            task='binary'
            ):

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    att_input = concat_func(list(chain.from_iterable(group_embedding_dict.values())), axis=1)

    for _ in range(att_layer_num):
        att_input = InteractingLayer(
            att_embedding_size, att_head_num, att_res)(att_input)
    att_output = Flatten()(att_input)

    dnn_input = combined_dnn_input(list(chain.from_iterable(group_embedding_dict.values())), dense_value_list)

    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    stack_out = Concatenate()([att_output, deep_out])
    dnn_logit = Dense(1, use_bias=False)(stack_out)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = Model(inputs=inputs_list, outputs=output)
    return model


