import sys
import yaml
 
import hourly_step
import weekly_step
 
GRAIN_RUNNERS = {
    "hourly": hourly_step.run,
    "weekly": weekly_step.run,
}
 
 
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
 
 
def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    grain = config["grain"]
    if grain not in GRAIN_RUNNERS:
        raise ValueError(f"Grain inconnu '{grain}' dans {config_path} -- attendu: {list(GRAIN_RUNNERS)}")
 
    print(f"=== Pipeline scarcity | grain={grain} | config={config_path} ===")
    return GRAIN_RUNNERS[grain](config)
 
 
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
 