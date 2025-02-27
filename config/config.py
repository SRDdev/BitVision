"""
This file contains the code to load the config.yaml file.
"""
import yaml

def load_config(path):
    """
    Load the config file
    """
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
        
if __name__ == "__main__":
    config = load_config("config/config.yaml")
    print(config['Architecture'])