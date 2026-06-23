from data.dataloder import get_loader
from utils import load_savemodel, generate_and_save
from models.network import Generator
import yaml

if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config.")

    save_path = config['testing']['save_path']
    img_size = config['testing']['img_size']
    device_ids = config['testing']['device_ids']
    current_epoch = config['testing']['current_epoch']
    test_output_dir = config['testing']['test_output_dir']

    base_model = Generator()
    model = load_savemodel(save_path, base_model, device_ids)

    test_loader, test_sampler = get_loader(
        root_A=config['data']['test_root_A'],
        root_B=config['data']['test_root_B'],
        mask_root_B=None,
        img_size=img_size,
        batch_size=1,
        is_train=False
    )

    # Execute the deterministic forward pass and persist the synthesized cross-modality images
    generate_and_save(current_epoch, test_loader, test_output_dir, model, device_ids)