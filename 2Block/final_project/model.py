import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv("merged.csv")
df = df.head(1000000)
# Определяем признаки
features = [col for col in df.columns if col not in ['target', 'user_id', 'post_id']]
X = df[features]
y = df['target']

# Разделим по пользователям
unique_users = df['user_id'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)

train_df = df[df['user_id'].isin(train_users)]
test_df = df[df['user_id'].isin(test_users)]

# Обучаем CatBoostClassifier
model = CatBoostClassifier(verbose=100, random_state=42)
model.fit(train_df[features], train_df['target'])

# Предсказание вероятностей
test_df['proba'] = model.predict_proba(test_df[features])[:, 1]

# HitRate@5
def hitrate_at_5(df):
    hits = 0
    users = df['user_id'].unique()
    for user in users:
        user_data = df[df['user_id'] == user]
        top5 = user_data.sort_values('proba', ascending=False).head(5)
        if top5['target'].sum() > 0:
            hits += 1
    return hits / len(users)

# Вывод метрики
hitrate = hitrate_at_5(test_df)
print(f"HitRate@5: {hitrate:.4f}")

# Сохранение модели
model.save_model("catboost_model.cbm")
