from configparser import ConfigParser
from pathlib import Path

config = ConfigParser()
config.read(f"{Path.home()}/.sortedness-cache.config")
try:  # pragma: no cover
    local_cache_uri = config.get("storages", "local")
    remote_cache_uri = config.get("storages", "remote")
except Exception as e:
    print(
        "Please create a config file '.sortedness-cache.config' in your home folder following the template file at the github repository."
    )
