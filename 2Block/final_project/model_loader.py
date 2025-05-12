from catboost import CatBoostClassifier
import os
import pandas as pd
from sqlalchemy import create_engine

import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime

app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True



def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        return '/workdir/user_input/model'
    else:
        return path

def load_models():
    model_path = get_model_path("catboost_model3.cbm")
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

model = load_models()
loaded_data = load_features()



def recommended_posts(user_id:int, timestamp:datetime, limit:int):
    user = loaded_data[0].loc[loaded_data[0].user_id == user_id].copy()
    user["x"]=1
    posts = loaded_data[1].copy()
    posts["x"]=1
    to_model = posts.merge(user, on='x',how='inner').set_index(['post_id',"user_id"]).drop(columns=['x','text','topic'])
    predictions = model.predict_proba(to_model)
    preds = pd.DataFrame(predictions[:,1], columns=['pred'])    
    posts_with_preds = pd.concat([posts, preds], axis=1).sort_values('pred', ascending=False)
    top5 = posts_with_preds[:limit][['post_id','text','topic']]
    top5_dict = top5.to_dict(orient='records')
    return top5_dict


import requests
from datetime import datetime

# Укажите URL вашего сервиса
url = "http://127.0.0.1:8000/post/recommendations/"

# Параметры запроса
params = {
    "id": 123,  # ID пользователя
    "time": datetime(2020, 5, 17), 
    "limit": 5
}

# Отправка GET-запроса
response = requests.get(url, params=params)

# Проверка статуса и вывод результата
if response.status_code == 200:
    recommendations = response.json()
    print("Рекомендованные посты:", recommendations)
else:
    print("Ошибка:", response.status_code, response.text)




@app.get("/post/recommendations/", response_model=List[PostGet])

def get_recommended_posts(id: int, timestamp:datetime, limit: int = 5) -> List[PostGet]:
    return recommended_posts(id, timestamp, limit)