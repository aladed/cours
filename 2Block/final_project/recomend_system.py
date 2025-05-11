import os
import pickle
import pandas as pd
from typing import List

from datetime import datetime
from pydantic import BaseModel
from sqlalchemy import create_engine
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_model():
    model_path = get_model_path("sklearn_model.pkl") # указать папку
    model = pickle.load(open(model_path, 'rb'))
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
    user_cf = batch_load_sql('SELECT * FROM bardin777_user_features_lesson_22')
    post_cf = batch_load_sql('SELECT * FROM bardin777_post_features_lesson_22')
    # likes = batch_load_sql("SELECT * FROM feed_data WHERE action = 'like' LIMIT 100000")
    return [user_cf, post_cf]

def recommended_posts(user_id:int, timestamp:datetime, limit:int):
    user = loaded_data[0].loc[loaded_data[0].user_id == user_id].copy()
    user["x"]=1
    posts = loaded_data[1].copy()
    posts["x"]=1
    date = pd.DataFrame({'day' : [timestamp.day], 'hour': [timestamp.hour], 'month': [timestamp.month], 'weekday': [timestamp.weekday()], 'x': 1})
    to_model = posts.merge(user, on='x',how='inner').merge(date, on='x',how='inner').set_index(['post_id',"user_id"]).drop(columns=['x','text','topic'])
    predictions = model.predict_proba(to_model)
    preds = pd.DataFrame(predictions[:,1], columns=['pred'])    
    posts_with_preds = pd.concat([posts, preds], axis=1).sort_values('pred', ascending=False)
    top5 = posts_with_preds[:limit][['post_id','text','topic']]
    top5_dict = top5.to_dict(orient='records')
    top5_objects = [PostGet(id=item['post_id'], text=item['text'], topic=item['topic']) for item in top5_dict]
    return top5_dict

model = load_model()
loaded_data = load_features()

@app.get("/post/recommendations/", response_model=List[PostGet])

def get_recommended_posts(id: int, timestamp:datetime, limit: int = 5) -> List[PostGet]:
    return recommended_posts(id, timestamp, limit)

