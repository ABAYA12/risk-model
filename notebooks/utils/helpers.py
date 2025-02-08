# utils/helpers.py
from sqlalchemy import create_engine
import pandas as pd

def load_data_from_db(table_name):
    # Replace with your actual database credentials
    DB_URL = "postgresql://postgres.njsusvrlnigkiduefyax:GfSNK6BnVuBUYEf7@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
    engine = create_engine(DB_URL)
    query = f'SELECT * FROM "{table_name}";'
    df = pd.read_sql(query, engine)
    return df