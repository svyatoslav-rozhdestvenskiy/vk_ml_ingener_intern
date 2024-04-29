import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import ndcg_score

df = pd.read_csv('datasets/intern_task.csv')
print(df.head(10))
print(df.shape)
print(df.isnull().sum().sum())
print(df[df.duplicated()])
print(df.dtypes.value_counts())
int_data = df.select_dtypes(include=['int64'])
print(int_data.head(10))
print(int_data.tail(10))
for column in int_data.columns:
    print(int_data[column].value_counts())
try:
    df.drop(int_data.columns[2:], axis=1, inplace=True)
except KeyError:
    print('Ошибка удаления данных, возможно вы удалили их ранее')
for column in df.columns:
    if column == 'rank' or column == 'query_id':
        continue
    if len(df[column].value_counts()) < 20:
        print(df[column].value_counts())


def show_plot(df, name_of_plot='dataframe'):
    min_seq = df.min()
    max_seq = df.max()
    mean_seq = df.mean()
    std_seq = df.std()
    plt.figure(figsize=(15, 40))
    plt.subplot(4, 1, 4)
    plt.scatter(min_seq.index, min_seq, label='min')
    plt.scatter(max_seq.index, max_seq, label='max')
    plt.scatter(mean_seq.index, mean_seq, label='mean')
    plt.scatter(std_seq.index, std_seq, label='std')
    plt.xlabel('feature')
    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.1, 0.1))
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 1)
    plt.scatter(min_seq.index, min_seq, label='min')
    plt.scatter(max_seq.index, max_seq, label='max')
    plt.scatter(mean_seq.index, mean_seq, label='mean')
    plt.scatter(std_seq.index, std_seq, label='std')
    plt.xlabel('feature')
    plt.yscale('symlog')
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.scatter(min_seq.index, min_seq, label='min')
    plt.scatter(max_seq.index, max_seq, label='max')
    plt.scatter(mean_seq.index, mean_seq, label='mean')
    plt.scatter(std_seq.index, std_seq, label='std')
    plt.xlabel('feature')
    plt.ylim(-10, 10)
    plt.yticks(np.arange(-10, 11, 1))
    plt.grid()
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.scatter(min_seq.index, min_seq, label='min')
    plt.scatter(max_seq.index, max_seq, label='max')
    plt.scatter(mean_seq.index, mean_seq, label='mean')
    plt.scatter(std_seq.index, std_seq, label='std')
    plt.xlabel('feature')
    plt.ylim(-500, 500)
    plt.yticks(np.arange(-500, 600, 100))
    plt.grid()
    plt.legend()
    plt.show()



show_plot(df=df[df.columns[2:]])
df_z = (df[df.columns[2:]] - df[df.columns[2:]].mean()) / df[df.columns[2:]].std()
df_z_med = (df[df.columns[2:]] - df[df.columns[2:]].median()) / df[df.columns[2:]].std()
df_norm = (df[df.columns[2:]] - df[df.columns[2:]].min()) / (df[df.columns[2:]].max() - df[df.columns[2:]].min())
show_plot(df_z, name_of_plot='z_normalization')
show_plot(df_z_med, name_of_plot='z_normalization_with_median')
show_plot(df_norm, name_of_plot='MinMax_normalization')
df_z_med = pd.concat([df[['rank', 'query_id']], df_z_med], axis=1)
train_queries, test_queries = train_test_split(df_z_med['query_id'].unique(), test_size=0.2, random_state=42)
x_train = df_z_med[df_z_med['query_id'].isin(train_queries)].drop(['rank', 'query_id'], axis=1)
y_train = df_z_med[df_z_med['query_id'].isin(train_queries)]['rank']
qid_train = df_z_med[df_z_med['query_id'].isin(train_queries)]
qid_train_list = qid_train.groupby(by='query_id').size().to_list()
x_test = df_z_med[df_z_med['query_id'].isin(test_queries)].drop(['rank', 'query_id'], axis=1)
y_test = df_z_med[df_z_med['query_id'].isin(test_queries)]['rank']
qid_test = df_z_med[df_z_med['query_id'].isin(test_queries)]
qid_test_list = qid_test.groupby(by='query_id').size().to_list()

model = xgb.XGBRanker(
    tree_method='hist',
    min_child_weight=0.1,
    eval_metric='ndcg',
    objective='rank:pairwise',
    random_state=42,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=110,
    early_stopping_rounds=10
    )

model.fit(x_train, y_train, group=qid_train_list, eval_group=[qid_test_list],
    eval_set=[(x_test, y_test)], verbose=True)
y_predict = model.predict(x_test)
print(y_predict)


def ndcg_grouped(y_true, y_pred, groups, k=5):
    start = 0
    ndcg_scores = []
    for group in groups:
        if group == 1:
            start += 1
            continue
        end = start + group
        ndcg = ndcg_score([y_true[start:end]], [y_pred[start:end]], k=k)
        ndcg_scores.append(ndcg)
        start = end
    return np.mean(ndcg_scores)

# Расчёт NDCG@5
ndcg_value = ndcg_grouped(y_test.to_numpy(), y_predict, qid_test_list, k=5)
print(f"NDCG@5 score: {ndcg_value}")