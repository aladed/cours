from catboost import CatBoostClassifier
import os
import pandas as pd
from sqlalchemy import create_engine


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    else:
        return path

def load_models():
    model_path = get_model_path("catboost_model")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)



def load_features() -> pd.DataFrame:
    user_cf = batch_load_sql('SELECT * FROM sivolapovvr_features_lesson_22')
    post_cf = batch_load_sql('SELECT * FROM sivolapovvr_post_features_lesson_22')
    return [user_cf, post_cf]

#тут мы должны брать все post_cf с user который дан в запросе и делаем по ним предикт


from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
from model_loader import load_models, load_features

# Инициализация FastAPI
app = FastAPI()

# 1. Загружаем модель и фичи один раз при старте
model = load_models()
user_df, post_df = load_features()

# 2. Эндпоинт рекомендаций
@app.get("/post/recommendations/")
def recommend(user_id: int):
    # Найдем фичи пользователя
    user_row = user_df[user_df["user_id"] == user_id]
    if user_row.empty:
        raise HTTPException(status_code=404, detail="User not found")

    # Повторим фичи пользователя для каждого поста
    user_expanded = pd.concat([user_row] * len(post_df), ignore_index=True).reset_index(drop=True)
    user_expanded = user_expanded.drop(columns=["user_id"], errors="ignore")  # убираем user_id из признаков

    # Объединяем с постами (axis=1 — по колонкам)
    predict_df = pd.concat([user_expanded, post_df.drop(columns=["post_id"], errors="ignore")], axis=1)

    # Добавляем обратно post_id для сортировки результатов
    predict_df["post_id"] = post_df["post_id"].values

    # Предсказание вероятностей
    probs = model.predict_proba(predict_df)[:, 1]
    predict_df["proba"] = probs

    # Возврат top-5 постов
    top_5_posts = predict_df.sort_values("proba", ascending=False).head(5)["post_id"].tolist()

    return {
        "user_id": user_id,
        "recommended_post_ids": top_5_posts
    }
