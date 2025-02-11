from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect

SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
