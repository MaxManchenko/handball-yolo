import yaml


def load_config(path_to_config_file: str = "config.yaml") -> dict:
    """Load and return the configuration data from the specified YAML file."""
    try:
        with open(path_to_config_file, "r", encoding="utf-8") as file:
            config_file = yaml.safe_load(file)
        return config_file
    except FileExistsError as err:
        raise FileExistsError("Configuration file not found.") from err
