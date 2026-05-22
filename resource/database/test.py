import pickle

# 使用 'rb' (read binary) 模式以二进制形式读取文件
with open('./bm25_corpus.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看数据的类型和具体内容
print("数据类型：", type(data))
print("数据内容：", data)