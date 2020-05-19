from pathlib import Path

import fire

from utils import SQLite, DB_NAME


def add_job(db, in_file, out_file, weight_file):
    in_file = Path(in_file).expanduser().resolve()
    out_file = Path(out_file).expanduser().resolve()
    weight_file = Path(weight_file).expanduser().resolve()

    if not all([in_file.is_file(), weight_file.is_file()]):
        in_file.is_file() or print(f"file {in_file} does not exist")
        weight_file.is_file() or print(f"file {weight_file} does not exist")
        exit(1)

    # Make output dir if doesn't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)

    db.execute(
        f"INSERT INTO jobs (infile,outfile,weightfile) "
        f"VALUES('{str(in_file)}','{str(out_file)}','{str(weight_file)}');")
    return db.commit()


def _add(in_file, out_file, weight_file):
    """
    Add a job to the processing queue.
    Args:
        in_file: A .tif file to detect kelp in.
        out_file: The desired .tif output file path with kelp labels.
        weight_file: The .pt weights file for the ML model to detect kelp.

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return add_job(db, in_file, out_file, weight_file)


def rm_job(db, *pks):
    for pk in pks:
        db.execute(f"DELETE FROM jobs WHERE pk={pk}")
    return db.commit()


def _rm(*pks):
    """
    Remove a job from the processing queue.
    Args:
        *pks: The list of job ids to remove, separated by spaces (e.g. 12 1 8 7)

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return rm_job(db, *pks)


def ls_jobs(db):
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


def _ls():
    """
    List all jobs in the queue and their status. Includes jobs with status "complete", "failed", and "scheduled".

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return ls_jobs(db)


def retry_job(db, *pks):
    for pk in pks:
        db.execute(f"UPDATE jobs SET status='scheduled' WHERE pk={pk}")
    return db.commit()


def _retry(*pks):
    """
    Reset jobs with id in *pks to "scheduled" so they are reprocessed.

    Args:
        *pks: The list of job ids to remove, separated by spaces (e.g. 12 1 8 7)

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return retry_job(db, *pks)


def complete_job(db, pk, failed=False):
    status = 'failed' if failed else 'complete'
    db.execute(f"UPDATE jobs SET status='{status}' WHERE pk={pk};")
    return db.commit()


def _complete(pk, failed=False):
    """
    Set job status to "complete" or "failed" using a list of ids.

    Args:
        *pks: The list of job ids to remove, separated by spaces (e.g. 12 1 8 7)
        failed: Boolean flag indicating to set the status to "failed" rather than the default "complete".

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return complete_job(db, pk, failed)


def ls_scheduled_jobs(db):
    curs = db.cursor()
    curs.execute("SELECT pk, infile, outfile, weightfile FROM jobs WHERE status='scheduled' ORDER BY created_dt;")
    return curs.fetchall()


def _ls_scheduled():
    """
    List all jobs in the queue where status is "scheduled".

    Returns: None
    """
    with SQLite(DB_NAME) as db:
        return ls_scheduled_jobs(db)


if __name__ == '__main__':
    fire.Fire({
        'add': _add,
        'rm': _rm,
        'ls': _ls,
        'retry': _retry,
        'ls_scheduled': _ls_scheduled,
        'complete': _complete
    })
