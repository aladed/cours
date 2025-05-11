from fastapi import FastAPI, HTTPException
import pandas as pd
import os
from catboost import CatBoostClassifier
from sqlalchemy import create_engine

# ========== Настройки ==========
MODEL_FILENAME = "improved_catboost_model.cbm"
USER_FEATURES_TABLE = "sivolapovvr_features_lesson_22"
POST_FEATURES_TABLE = "sivolapovvr_post_features_lesson_22"

# ========== Загрузка модели ==========
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    return path

def load_model() -> CatBoostClassifier:
    model_path = get_model_path(MODEL_FILENAME)
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

# ========== Загрузка фичей из БД ==========
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200_000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features():
    user_df = batch_load_sql(f"SELECT * FROM {USER_FEATURES_TABLE}")
    post_df = batch_load_sql(f"SELECT * FROM {POST_FEATURES_TABLE}")
    return user_df, post_df

# ========== Инициализация ==========
app = FastAPI()

model = load_model()
user_df, post_df = load_features()

# ========== Эндпоинт рекомендаций ==========
@app.get("/post/recommendations/")
def recommend(user_id: int):
    # Проверка: есть ли user
    user_row = user_df[user_df["user_id"] == user_id]
    if user_row.empty:
        raise HTTPException(status_code=404, detail="User not found")

    # Дублируем user-фичи на все посты
    user_feats = pd.concat([user_row] * len(post_df), ignore_index=True).drop(columns=["user_id"], errors="ignore")

    # Объединяем с постовыми фичами
    df_pred = pd.concat([user_feats, post_df.drop(columns=["post_id"], errors="ignore")], axis=1)
    df_pred["post_id"] = post_df["post_id"].values

    # Предикт
    df_pred["proba"] = model.predict_proba(df_pred)[:, 1]

    # Топ-5 постов
    top5 = df_pred.sort_values("proba", ascending=False).head(5)["post_id"].tolist()

    return {
        "user_id": user_id,
        "recommended_post_ids": top5
    }

