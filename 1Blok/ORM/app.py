from fastapi import Depends, FastAPI,HTTPException
from sqlalchemy.orm import Session
from typing import List
from sqlalchemy.sql.functions import count

from database import SessionLocal
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import  UserGet,PostGet,FeedGet

from typing import Optional


app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db

@app.get("/user/{id}", response_model= UserGet)
def get_user(id: int , db: Session = Depends(get_db)):
    result = db.query(User).filter(User.id==id).one_or_none()
    if not result:
        raise HTTPException(404, "user not found")
    else:
        return result

@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int , db: Session = Depends(get_db)):
    result = db.query(Post).filter(Post.id==id).one_or_none()
    if not result:
        raise HTTPException(404, "post not found")
    else:
        return result
    


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):
    results = (
        db.query(Feed)
        .filter(Feed.user_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    if not results:
        raise HTTPException(404, "not found")
    else:
        return results



@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):
    results = (
        db.query(Feed)
        .filter(Feed.post_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    if not results:
        raise HTTPException(404, "not found")
    else:
        return results


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(id: Optional[int] = None, limit: int = 10, db: Session = Depends(get_db)):
    top_posts = (
        db.query(Post)
        .select_from(Feed)
        .filter(Feed.action == "like")
        .join(Post, Post.id == Feed.post_id)
        .group_by(Post.id)
        .order_by(count(Feed.post_id).desc())
        .limit(limit)
        .all()
    )
    if not top_posts:
        raise HTTPException(404, "not found")
    else:
        return top_posts