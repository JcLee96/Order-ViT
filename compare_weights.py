from pytorch_lightning import LightningModule
import timm
import torch
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def compare_weights(config):
    # if not os.path.isabs(config.ckpt_path):
    #     config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.ckpt_path)

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(
        '/path/to/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt')
    # torch.save(model, '/path/to/jh/project/colon_compare/model.pt')
    pretrained_model = timm.create_model('vit_large_r50_s32_384', pretrained=True, num_classes=4)
    all_layer = 0
    diff_layer = 0
    for (n1, w1), (n2, w2) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        all_layer += 1

        if n2 in 'head':
            print('now stage is head')

        if (w1.data - w2.data).abs().sum() == 0:
            print(f'In {n1}, {n2} It is same')
        else:
            diff_layer += 1
            print(n1)
    print(f'all_layer: {all_layer}\n'
          f'different layer is {diff_layer}')


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return compare_weights(config)


if __name__ == "__main__":
    main()
