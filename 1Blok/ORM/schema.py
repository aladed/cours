import datetime as dt
from pydantic import BaseModel, Field


class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str=""
    city: str=""
    exp_group: int
    os: str=""
    source: str=""

    class Config:
        orm_mode = True

class PostGet(BaseModel):
    id: int
    text: str=""
    topic: str=""

    class Config:
        orm_mode = True

class FeedGet(BaseModel):
    user_id: int
    user: UserGet
    post_id: int
    post: PostGet
    action: str
    time: dt.datetime
    
    class Config:
        orm_mode = True