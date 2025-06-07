import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from model.mind import MIND
from model.match_trainer import MatchTrainer
from model.basic.features import DenseFeature, SparseFeature, SequenceFeature
from model.utils.match import generate_seq_feature_match, gen_model_input
from model.utils.data import df_to_dict, MatchDataGenerator
from model.utils.movielens_utils import match_evaluation
import data_processing
import faiss

pd.set_option('display.max_columns', None)  # 显示所有列
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(proj_path):
    model_path = os.path.join(proj_path,'examples/data/ml-1m/saved/model_5.pth')
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
    history_features = [SequenceFeature('hist_movie_id',vocab_size=feature_max_idx['movie_id'],embed_dim=16,pooling='concat',shared_with="movie_id")]
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
    model.load_state_dict(torch.load(model_path))
    return model

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

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """ Pads sequences (list of list) to the ndarray of same length.
        This is an equivalent implementation of tf.keras.preprocessing.sequence.pad_sequences
        reference: benchmark/FuxiCTR/fuxictr at main · huawei-noah/benchmark

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

##存储数据到向量数据库
def save_items_embedding_to_faiss(proj_path,model):
    dim = 16  # 向量维度
    index_faiss = faiss.IndexFlatIP(dim)
    index_id = faiss.IndexIDMap(index_faiss)
    data_dir = os.path.join(proj_path, 'data')
    data_dict = data_processing.load_and_preprocess_data(data_dir)
    data = data_dict['df']

    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    sparse_features = ['movie_id','cate_id']
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature+'_encode'] = lbe.fit_transform(data[feature]) + 1
    movies_cate_mapping_df = data[['movie_id','cate_id','movie_id_encode','cate_id_encode']].drop_duplicates()
    print(movies_cate_mapping_df.shape)
    movies_cate_mapping_df.to_csv('./data/ml-1m/saved/movies_index_map.csv')
    movies_info=[]
    movies_ids = []
    for index, row in movies_cate_mapping_df.iterrows():
        item_dict={}
        item_dict['user_id'] = 0
        item_dict['gender'] = 0
        item_dict['age'] = 0
        item_dict['city'] = 0
        item_dict['occupation'] = 0
        item_dict['zip_code'] = 0
        item_dict['hist_movie_id'] = [[0]],
        item_dict['histlen_movie_id'] = 0
        item_dict['movie_id'] = row['movie_id_encode']
        item_dict['cate_id'] = row['cate_id_encode']
        item_dict['neg_items'] = [[0]]
        movies_info.append(item_dict)
        movies_ids.append(row['movie_id'])
    x_dict_list = []
    for item in movies_info:
        x_dict_list.append(get_model_input(item))
    model.mode='item'
    x_data_embedding = get_inference_embedding(model, x_dict_list)
    database = x_data_embedding.numpy()
    # print(database)
    # index.add(database)
    # index_faiss.add(database)
    index_id.add_with_ids(database, movies_ids)
    faiss.write_index(index_faiss, "./data/ml-1m/saved/index_id_faiss.faiss")


def generate_numbers(target_N):
    import random
    numbers = [target_N//4 for _ in range(4)]  # 先生成3个随机数，最后一个由前三个决定
    total = sum(numbers)
    last_number = target_N - total  # 计算最后一个数的值使其总和为10
    numbers[-1] += last_number
    return numbers

##查询数据到向量数据库，返回10个movie_ids
def query_by_embedding(user_data_emb,proj_path,topN=10):
    faiss_data_path = os.path.join(proj_path,'examples/data/ml-1m/saved/index_id_faiss.faiss')
    index_loaded = faiss.read_index(faiss_data_path)
    print("evaluate embedding matching on test data")
    indices_list = []
    distances_list = []
    print(user_data_emb.shape)
    numbers = generate_numbers(topN)
    for i in range(user_data_emb.shape[1]):
        query_emb = np.expand_dims(user_data_emb[0][i],axis=0)
        distances, indices = index_loaded.search(query_emb,numbers[i])
        indices_list += indices[0].tolist()
        distances_list += distances[0].tolist()
    print("返回的自定义id:", indices_list)  # 应该是10
    print("返回的自定义id的距离",distances_list)
    return distances_list,indices_list




if __name__ == '__main__':
    current_pth = os.getcwd()
    proj_path = os.path.dirname(current_pth)
    print(proj_path)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_path', default=proj_path)
    args = parser.parse_args()

    print(args)
    model = load_model(args.proj_path)
    ## 存储向量数据
    # save_items_embedding_to_faiss(args.proj_path, model)
    # 假设单用户数据 (已预处理)
    users_data = [{
        "user_id":1223,
        "gender": 2,
        "age": 0.3,  # 归一化后的值
        "city": 15,
        "occupation":8,
        "zip_code": 67,
        "hist_movie_id":[[3218, 848,1901, 3619, 2384, 837, 1442, 2324, 348]],
        "histlen_movie_id":7,  # 补齐后的序列
        "movie_id": 10,  # 待预测物品
        "cate_id": 10,
        "neg_items": [[2954, 2676, 1]]
    }]

    x_dict_list = []
    for item in users_data:
        x_dict_list.append(get_model_input(item))
    # print(x_dict_list)
    model.mode = 'user'
    users_data_embedding = get_inference_embedding(model, x_dict_list)
    users_data_embed =users_data_embedding.numpy()
    print(users_data_embed)
    print(users_data_embed.shape)
    query_by_embedding(users_data_embed,args.proj_path)


# 返回的自定义id: [1901, 3619, 1901, 3619, 1901, 3619, 1901, 3619, 2384, 837]
# 返回的自定义id的距离 [0.08207644522190094, 0.04839305579662323, 0.08252008259296417, 0.05004461109638214, 0.08227398991584778, 0.04960280656814575, 0.08226533234119415, 0.04976564645767212, 0.03293493390083313, 0.016667142510414124]