from database import Base, SessionLocal
from sqlalchemy import Column, String, Integer
from sqlalchemy import func, select


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


if __name__ == "__main__":
    session = SessionLocal()
    result = [
        post.id for post in (
            session.query(Post)
            .filter(Post.topic == "business")
            .order_by(Post.id.desc()).limit(10)
            .all()
        )
    ]

    print(result)