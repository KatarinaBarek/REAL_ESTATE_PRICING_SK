import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL

load_dotenv()

DB_URL = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 3306)),
    database=os.getenv("DB_NAME"),
)

TABLE_NAME = "final_flat_listings"
TARGET_COL = "price_current"
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
