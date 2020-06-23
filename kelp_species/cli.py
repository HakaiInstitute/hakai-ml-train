import fire

from kelp_species.predictor import predict
from kelp_species.trainer import train

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'pred': predict
    })
