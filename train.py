import torch.multiprocessing as mp
import os
import yaml
from utils import set_seed, setup, cleanup
from models.trainer import Trainer

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main_worker(rank, world_size, config):
    setup(rank, world_size)
    set_seed(42)
    try:
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    finally:
        cleanup()

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config.")

    world_size = 4

    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
