import fire

from seagrass.predictor import predict
from seagrass.trainer import train

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'pred': predict
    })
