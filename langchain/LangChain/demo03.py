from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.columns.tolist())
engine = create_engine("sqlite:///titanic.db")
df.to_sql("titanic", engine, index=False)

