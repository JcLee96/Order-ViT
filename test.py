import dotenv
import hydra
from omegaconf import DictConfig
import warnings, os
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.testing_pipeline import test


    os.environ["WANDB_API_KEY"] = 'wandb-key-number'

    # Applies optional utilities
    utils.extras(config)
    warnings.filterwarnings(action='ignore')

    # Evaluate model
    return test(config)


if __name__ == "__main__":
    main()
