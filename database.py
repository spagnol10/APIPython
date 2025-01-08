from databases import Database
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://udbfacialhom:Dw^X3vpMel30HO%S@dbdevprisma.postgres.database.azure.com:5432/dbfacial_hom")

database = Database(DATABASE_URL)