from configparser import ConfigParser
from pathlib import Path


config = ConfigParser()
config.read(f"{Path.home()}/.sortedness-cache.config")
local_cache_uri = config.get("storages", "local")
remote_cache_uri = config.get("storages", "remote")
