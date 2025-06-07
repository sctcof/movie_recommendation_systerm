import os
from random import randint

import torch

from model.mind import MIND
import numpy as np
from model.basic.features import DenseFeature, SparseFeature, SequenceFeature
import data_processing
import pandas as pd
import faiss
from collections import OrderedDict, Counter
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
pd.set_option('display.max_columns', None)  # 显示所有列
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def negative_sample(items_cnt_order, ratio, method_id=0):
    """Negative Sample method for matching model
    reference: https://github.com/wangzhegeek/DSSM-Lookalike/blob/master/utils.py
    update more method and redesign this function.

    Args:
        items_cnt_order (dict): the item count dict, the keys(item) sorted by value(count) in reverse order.
        ratio (int): negative sample ratio, >= 1
        method_id (int, optional):
        `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.

    Returns:
        list: sampled negative item list
    """
    items_set = [item for item, count in items_cnt_order.items()]
    if method_id == 0:
        neg_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method_id == 1:
        # items_cnt_freq = {item: count/len(items_cnt) for item, count in items_cnt_order.items()}
        # p_sel = {item: np.sqrt(1e-5/items_cnt_freq[item]) for item in items_cnt_order}
        # The most popular paramter is item_cnt**0.75:
        p_sel = {item: count ** 0.75 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 2:
        p_sel = {item: np.log(count + 1) + 1e-6 for item, count in items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method_id == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1)) / np.log(len(items_cnt_order) + 1) for item, k in
                 items_cnt_order.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        neg_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    else:
        raise ValueError("method id should in (0,1,2,3)")
    return neg_items

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: https://github.com/huawei-noah/benchmark/tree/main/FuxiCTR/fuxictr

    Args:
        sequences (pd.DataFrame): data that needs to pad or truncate
        maxlen (int): maximum sequence length. Defaults to None.
        dtype (str, optional): Defaults to 'int32'.
        padding (str, optional): if len(sequences) less than maxlen, padding style, {'pre', 'post'}. Defaults to 'pre'.
        truncating (str, optional): if len(sequences) more than maxlen, truncate style, {'pre', 'post'}. Defaults to 'pre'.
        value (_type_, optional): Defaults to 0..

    Returns:
        _type_: _description_
    """

    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncating in ["pre", "post"], "Invalid truncating={}.".format(truncating)

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        if truncating == 'pre':
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr

def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict

def get_model_input(user_data,seq_max_len=50,padding='post', truncating='post'):
    user_df = pd.DataFrame(user_data)
    for col in user_df.columns.to_list():
        if col.startswith("hist_"):
            user_df[col] = pad_sequences(user_df[col], maxlen=seq_max_len, value=0, padding=padding,
                                    truncating=truncating).tolist()
    user_df_dict = df_to_dict(user_df)
    x_dict = {k: torch.tensor(v).to(device) for k, v in user_df_dict.items()}
    return x_dict


def get_inference_embedding(model, x_dict_list):
    # inference
    model = model.to(device)
    model.eval()
    predicts = []
    with torch.no_grad():
        for x_dict in x_dict_list:
            y_pred = model(x_dict)
            predicts.append(y_pred.data)
    return torch.cat(predicts, dim=0)

def load_mind_model(proj_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(proj_path, 'examples/data/ml-1m/saved/model_5.pth')
    # 定义特征
    feature_max_idx = {}
    feature_max_idx['user_id'] = 6041
    feature_max_idx['gender'] = 3
    feature_max_idx['age'] = 8
    feature_max_idx['occupation'] = 22
    feature_max_idx['zip_code'] = 3440
    feature_max_idx['movie_id'] = 3707
    feature_max_idx['cate_id'] = 19
    user_cols = ['gender', 'age', 'occupation', 'zip_code']
    user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
    history_features = [
        SequenceFeature('hist_movie_id', vocab_size=feature_max_idx['movie_id'], embed_dim=16, pooling='concat',
                        shared_with="movie_id")]
    item_cols = ['movie_id', 'cate_id']
    item_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in item_cols]
    neg_item_feature = [
        SequenceFeature('neg_items',
                        vocab_size=feature_max_idx['movie_id'],
                        embed_dim=16,
                        pooling="concat",
                        shared_with="movie_id")
    ]
    seq_max_len = 50
    model = MIND(user_features, history_features, item_features, None, max_length=seq_max_len,
                 temperature=0.02)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def get_user_ebebdding_by_user_id(user_id,model,data_path,sample_method=1,neg_ratio=3):
    data_dir = os.path.join(data_path, 'data')
    data_dict = data_processing.load_and_preprocess_data(data_dir)
    data_df = data_dict['df']
    data_df["cate_id"] = data_df["genres"].apply(lambda x: x.split("|")[0])
    data_df["label"] = data_df["rating"].apply(lambda x: 1 if x >= 3 else 0)
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code', "cate_id"]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data_df[feature] = lbe.fit_transform(data_df[feature]) + 1
    print('user_id:',user_id)
    data_df['match']=data_df['user_id'].apply(lambda x: x == user_id)
    data = data_df[data_df['match']==True]
    user_col, item_col, label_col = "user_id", "movie_id", "label"
    items_cnt = Counter(data[item_col].tolist())  # item的量级
    items_cnt_order = OrderedDict(sorted((items_cnt.items()), key=lambda x: x[1], reverse=True))  # item_id:item count
    neg_list = negative_sample(items_cnt_order, ratio=data.shape[0] * neg_ratio, method_id=sample_method)
    time_col = 'timestamp'
    data.sort_values(time_col,ascending=False, inplace=True)  # sort by time from old to new
    # print(data.head())
    pos_list=[]
    for uid,hist in data.groupby(user_col):
        pos_list = hist[item_col].tolist()
    index_it = randint(0,min(20,len(pos_list)))
    hist_movie_id_list=pos_list[:index_it]
    hist_movie_id_len = len(hist_movie_id_list)
    # print(hist_movie_id_list)
    user_info_df = data.iloc[0]
    # print(user_info_df)
    users_data = [{
        "user_id": user_info_df['user_id'],
        "gender": user_info_df['gender'],
        "age":user_info_df['age'],  # 归一化后的值
        "occupation": user_info_df['occupation'],
        "zip_code": user_info_df['zip_code'],
        "hist_movie_id": [hist_movie_id_list],
        "histlen_movie_id": 0,  # 补齐后的序列
        "movie_id": 0,  # 待预测物品
        "cate_id": 0,
        "neg_items": [neg_list[0:4]]
    }]
    # print(users_data)
    x_dict_list = []
    for item in users_data:
        x_dict_list.append(get_model_input(item))
    # print(x_dict_list)
    model.mode='user'
    users_data_embedding = get_inference_embedding(model, x_dict_list)
    users_data_embed = users_data_embedding.numpy()
    # print(users_data_embed)
    return users_data_embed

##查询数据到向量数据库，返回10个movie_ids
def generate_numbers(target_N):
    import random
    numbers = [target_N//4 for _ in range(4)]  # 先生成3个随机数，最后一个由前三个决定
    total = sum(numbers)
    last_number = target_N - total  # 计算最后一个数的值使其总和为10
    numbers[-1] += last_number
    return numbers

##查询数据到向量数据库，返回10个movie_ids
def query_by_embedding(user_data_emb,topN=10):
    faiss_data_path = os.path.join(os.getcwd(),'examples/data/ml-1m/saved/index_id_faiss.faiss')
    index_loaded = faiss.read_index(faiss_data_path)
    print("evaluate embedding matching on test data")
    print(user_data_emb.shape)
    user_data_emb = np.squeeze(user_data_emb,0)
    user_data_emb = np.mean(user_data_emb,axis=0)
    query_emb = np.expand_dims(user_data_emb, axis=0)
    distances, indices = index_loaded.search(query_emb, topN)
    print("返回的自定义id:", indices[0])  # 应该是10
    print("返回的自定义id的距离", distances[0])
    return distances[0], indices[0]

    # indices_list = []
    # distances_list = []
    # #存在重复的情况
    # numbers = generate_numbers(topN)
    # for i in range(user_data_emb.shape[1]):
    #     query_emb = np.expand_dims(user_data_emb[0][i],axis=0)
    #     distances, indices = index_loaded.search(query_emb,numbers[i])
    #     indices_set.add(indices[0].tolist())
    #     indices_list += indices[0].tolist()
    #     distances_list += distances[0].tolist()
    # print("返回的自定义id:", indices_list)  # 应该是10
    # print("返回的自定义id的距离",distances_list)
    # return distances_list,indices_list

#
# # # def get_movie_info_by(movie_id):
# proj_path = '/Users/ruiqing/PycharmProjects/movie_recommendation_system'
# model = load_mind_model(proj_path)
# embedd1 = get_user_ebebdding_by_user_id(4562,model,proj_path)
# print(embedd1)
# query_by_embedding(embedd1,proj_path,20)