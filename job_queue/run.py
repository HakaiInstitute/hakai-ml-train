import os
import subprocess
from collections import deque
from pathlib import Path
from time import sleep

from job import complete_job, ls_scheduled_jobs

from utils import SQLite, DB_NAME


def job_loop(db):
    q = deque()

    while True:
        if len(q) == 0:
            # Add items to q
            new_jobs = ls_scheduled_jobs(db)
            if len(new_jobs) > 0:
                for job in new_jobs:
                    q.append(job)
            else:
                sleep(60)  # Wait a minute

        else:
            # Process jobs using docker script
            pk, in_file, out_file, weight_file = q.popleft()
            print(f"Running job: {pk}")

            if not all([Path(f).is_file() for f in [in_file, weight_file]]):
                print(f"Invalid path specified for job {pk}")
                complete_job(db, pk, failed=True)
                continue

            if not Path(out_file).parent.is_dir():
                Path(out_file).parent.mkdir(parents=True, exist_ok=True)

            subprocess.run([
                "docker", "run", "--rm",
                f"-v", f"{in_file}:{in_file}:ro",
                f"-v", f"{Path(weight_file).parent}:{Path(weight_file).parent}:ro",
                "-v", f"{Path(out_file).parent}:{Path(out_file).parent}",
                f"--user={os.geteuid()}:{os.getgid()}",
                "--ipc=host",
                "--gpus=all",
                "--name=kelp-pred",
                "tayden/deeplabv3-kelp", "pred",
                f"--seg_in={in_file}", f"--seg_out={out_file}", f"--weights={weight_file}"
            ], cwd='../', check=True)

            # Mark job complete
            complete_job(db, pk)


if __name__ == '__main__':
    with SQLite(DB_NAME) as db:
        job_loop(db)
