from pathlib import Path

import fire

from utils import SQLite


def add_job(in_file, out_file, weight_file):
    in_file = Path(in_file).expanduser().resolve()
    out_file = Path(out_file).expanduser().resolve()
    weight_file = Path(weight_file).expanduser().resolve()

    for f in [in_file, out_file, weight_file]:
        if not f.is_file():
            print(f"file {f} does not exist")

    with SQLite("kelp.sqlite") as db:
        db.execute(
            f"INSERT INTO jobs (infile,outfile,weightfile) "
            f"VALUES('{str(in_file)}','{str(out_file)}','{str(weight_file)}');")
        return db.commit()


def rm_job(pk):
    with SQLite("kelp.sqlite") as db:
        db.execute(f"DELETE FROM jobs WHERE pk={pk}")
        return db.commit()


def ls_jobs():
    with SQLite("kelp.sqlite") as db:
        curs = db.cursor()
        curs.execute(f"SELECT pk, status, created_dt, infile, outfile, weightfile FROM jobs")
        return [{
            "pk": j[0],
            "status": j[1],
            "submitted": j[2],
            "in_file": j[3],
            "out_file": j[4],
            "weights": j[5],
        } for j in curs.fetchall()]


if __name__ == '__main__':
    fire.Fire({
        'add': add_job,
        'rm': rm_job,
        'ls': ls_jobs
    })
