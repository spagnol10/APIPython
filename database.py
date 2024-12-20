from databases import Database
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/postgres")

# Conexão assíncrona ao banco de dados
database = Database(DATABASE_URL)
