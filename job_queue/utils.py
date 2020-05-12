from sqlite3 import connect


class SQLite(object):
    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        self.db = connect(self.db_path)
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()