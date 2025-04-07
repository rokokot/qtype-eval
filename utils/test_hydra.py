import hydra
from omegaconf import DictConfig

@hydra.main(config_path=None)
def main(cfg: DictConfig) -> None:
    print("Hydra is working!")
    print(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

if __name__ == "__main__":
    main()
