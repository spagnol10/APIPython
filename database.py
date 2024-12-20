from databases import Database
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root@localhost:5432/postgres")

database = Database(DATABASE_URL)