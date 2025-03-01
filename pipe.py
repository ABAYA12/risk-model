import psycopg2 as pg
import os

# dbname = os.getenv('database_name')
# user = os.getenv('user')
# password = os.getenv('password')
# host = os.getenv('host')
# port = os.getenv('port')
host = "db.fsmnqzkdkecuinjukcqo.supabase.co"
port = 5432
dbname = "postgres"
user = "postgres"
password = "RiskGuard AI"
sslmode = "require"  # Ensure SSL is enabled

conn = pg.connect(
    dbname=dbname, user=user, password=password, host=host, port=port, options="-c inet6=off"
)

cur = conn.cursor()
query = "SELECT * FROM risk_facts"
cur.execute(query)
results = cur.fetchall()
print(results)
cur.close()

