# --*-- conding:utf-8 --*--
# @Time : 3/18/25 1:41 AM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : config_manager.py

class ConfigManager:
    def __init__(self, config_path):
        self.config_dict = self._read_config(config_path)

    def _read_config(self, path):
        config = {}
        with open(path, 'r') as f:
            for line in f:
                if "=" in line:
                    k,v = line.strip().split("=")
                    config[k.strip()] = v.strip()
        return config

    def get(self, key, default=None):
        return self.config_dict.get(key, default)
