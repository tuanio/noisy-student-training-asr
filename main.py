import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    ...


if __name__ == "__main__":
    main()
