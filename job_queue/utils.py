import subprocess
from sqlite3 import connect
from pathlib import Path


class SQLite(object):
    def __init__(self, db_path):
        self.db_path = db_path
        if not Path(db_path).expanduser().resolve().is_file():
            self._make_db(self.db_path)

    def __enter__(self):
        self.db = connect(self.db_path)
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    @staticmethod
    def _make_db(db_name):
        subprocess.run(f"sqlite3 {db_name} < create_jobs_db.sql", shell=True, check=True)
