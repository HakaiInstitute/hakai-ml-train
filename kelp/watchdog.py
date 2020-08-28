"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-26
Description: 
"""
import time
from pathlib import Path

import fire
from loguru import logger

from kelp.predictor import predict

SLEEP_T = 5
MATCH_PATTERN = "**/*.tif"
DEFAULT_PRESENCE_WEIGHTS = "https://hakai-deep-learning-datasets.s3.amazonaws.com" \
                           "/kelp/weights/deeplabv3_kelp_200704.ckpt"
DEFAULT_SPECIES_WEIGHTS = "https://hakai-deep-learning-datasets.s3.amazonaws.com" \
                          "/kelp_species/weights/deeplabv3_kelp_species_200818.ckpt"


@logger.catch
def segment_kelp(in_path: str, out_dir: str, weights_path: str) -> str:
    out_path = Path(out_dir).joinpath(f"{Path(in_path).stem}_kelp.tif")
    predict(in_path, out_path, weights_path)
    return out_path


@logger.catch
def watch_dir(in_dir: str, out_dir: str):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    while True:
        files = list(in_dir.glob(MATCH_PATTERN))

        if len(files):
            next_path = files[0]
            logger.info(f"Processing file {str(next_path)}")

            out_loc = out_dir.joinpath(next_path.relative_to(in_dir).parent)

            weights_path = None

            job_type = next_path.relative_to(in_dir).parts[0]
            if job_type and job_type == 'presence':
                weights_path = DEFAULT_PRESENCE_WEIGHTS
            elif job_type and job_type == 'species':
                weights_path = DEFAULT_SPECIES_WEIGHTS

            if not weights_path:
                logger.error(f"No weights associated with directory {job_type}. Skipping")
            else:
                # Make the kelp output file
                output_kelp_path = segment_kelp(str(next_path), out_loc, weights_path=weights_path)
                logger.info(f"File {output_kelp_path} created")

            # Move the input file(s) to the out_dir
            for p in next_path.parent.glob(f"{next_path.stem}.*"):
                move_to_path = Path(out_loc).joinpath(p.name)
                p.rename(move_to_path)
                logger.info(f"Moved {str(p)} to {str(move_to_path)}")

        else:
            time.sleep(SLEEP_T)


if __name__ == '__main__':
    logger.add("./watchdog.log", rotation="100MB", retention="3 months", compression="gz", enqueue=True)
    fire.Fire(watch_dir)
