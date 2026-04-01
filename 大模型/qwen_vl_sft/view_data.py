import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/Users/guangshengliu/Desktop/code/RAG/sft/data/train-00000-of-00001.parquet') 

# 显示前几行
print(df.head())