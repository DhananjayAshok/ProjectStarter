import json
import os
import yaml
import datetime
from .log_handling import log_error
import hashlib

def write_meta(write_dir, meta_dict, logger):
    meta_hash = hash_meta_dict(meta_dict)
    meta_dict["write_timestamp"] = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    meta_path = os.path.join(write_dir, f"meta_{meta_hash}.yaml")
    if not os.path.exists(write_dir):
        log_error(logger, f"Directory {write_dir} does not exist for meta file writing")
    with open(meta_path, "w") as f:
        yaml.dump(meta_dict, f)
    return meta_hash

def meta_dict_to_str(meta_dict, print_mode=False, n_indents=1, skip_write_timestamp=True):
    keys = list(meta_dict.keys())
    keys.sort()
    meta_str = ""
    for key in keys:
        if print_mode:
            indent = '\t' * n_indents
            meta_str += f"{indent}{key}: {meta_dict[key]}\n"
        else:
            if skip_write_timestamp and key == "write_timestamp":
                continue
            meta_str += f"{key.lower().strip()}_{str(meta_dict[key]).lower().strip()}"
    return meta_str

def hash_meta_dict(meta_dict):
    meta_str = meta_dict_to_str(meta_dict)
    hash_obj = hashlib.sha256(meta_str.encode('utf-8'))
    hex_hash = hash_obj.hexdigest()[:10] # guessing that this won't cause problems. It could though, lol
    return hex_hash

def add_meta_details(meta_dict, add_details):
    alt = meta_dict.copy()
    alt.update(add_details)
    return alt
