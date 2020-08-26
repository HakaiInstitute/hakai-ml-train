"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-26
Description: 
"""
import time
from pathlib import Path

import fire
import requests
from loguru import logger
from tqdm import tqdm

from kelp.predictor import predict

SLEEP_T = 5
MATCH_PATTERN = "**/*.tif"
DEFAULT_WEIGHTS_LOCAL_PATH = "./presence/weights/deeplabv3_kelp_200704.ckpt"
DEFAULT_WEIGHTS_S3_PATH = "https://hakai-deep-learning-datasets.s3.amazonaws.com" \
                          "/kelp/weights/deeplabv3_kelp_200704.ckpt"
ONE_MB = 2 << 20  # Bytes


@logger.catch
def download_weights(s3_path, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(s3_path, stream=True) as r:
        r.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=ONE_MB), desc="Downloading weights"):
                f.write(chunk)


@logger.catch
def segment_kelp_presence(in_path: str, out_dir: str, weights_path: str = DEFAULT_WEIGHTS_LOCAL_PATH) -> str:
    in_path = Path(in_path)
    out_dir = Path(out_dir)

    if not Path(weights_path).is_file():
        if weights_path != DEFAULT_WEIGHTS_LOCAL_PATH:
            logger.warning(f"Weights {weights_path} not available, using default {DEFAULT_WEIGHTS_LOCAL_PATH} instead")
        download_weights(DEFAULT_WEIGHTS_S3_PATH, DEFAULT_WEIGHTS_LOCAL_PATH)
        weights_path = DEFAULT_WEIGHTS_LOCAL_PATH

    out_path = out_dir.joinpath(f"{in_path.stem}_kelp.tif")

    predict(str(in_path), str(out_path), weights_path)

    return str(out_path)


@logger.catch
def watch_dir(in_dir: str, out_dir: str):
    in_dir = Path(in_dir)
    while True:
        files = list(in_dir.glob(MATCH_PATTERN))

        if len(files):
            next_path = files[0]
            logger.info(f"Processing file {str(next_path)}")

            # Make the kelp output file
            output_kelp_path = segment_kelp_presence(str(next_path), out_dir)
            logger.info(f"File {output_kelp_path} created")

            # Move the file to the out_dir
            move_to_path = Path(out_dir).joinpath(next_path.name)
            next_path.rename(move_to_path)
            logger.info(f"Moved {str(next_path)} to {str(move_to_path)}")

        else:
            time.sleep(SLEEP_T)


if __name__ == '__main__':
    logger.add("./watchdog.log", rotation="100MB", retention="3 months", compression="gz", enqueue=True)
    fire.Fire(watch_dir)
