# Project Starter
## Overview

This project lets you easily set up:
* Standardised environment set up with `uv`
* Large file sharing with [HuggingFace Datasets](https://huggingface.co/docs/datasets/en/index)
* Relatively clean parameter handling; add your parameters to a single config file, and you get the ability to access them anywhere you want.

Core functionality:
| Function(s) / Class| Description |  Common Pattern
| - | - |  -
| `load_parameters` | Used to read the parameters from the config files and return a populated dictionary. You can also pass in an existing parameters dictionary, in which case this dict will just be returned | `parameters = load_parameters(parameters)`
| `log_error`, `log_warn`, `log_info`, `log_dict` | Used to send specific messages to the logger. Takes an optional parameter argument that matters if and only if you pass in a `--log_file` argument | `log_warn('Bruh')` if you do not intend on passing in a log_file, `log_warn('Bruh', parameters)` if you do|
|`sync_data.py` | Used to sync files in the storage_dir/sync directories with the HuggingFace Datasets repo | `python sync_data.py pull`, `python sync_data.py push`, `python sync_data.py init`|

Secondary functionality:

| Function(s) / Class| Description |  Common Pattern
| - | - |  -
|`write_meta` |Used to save a dictionary of arguments to a specified path. This hashes the dict based on its argument values, allowing you to create a unique identifier for a particular configuration of a run. Should be used when you create an artifact (e.g. a model) that has hyperparameters you want to keep track of| `args={'random_seed': 42}` and then `write_meta('data_process/', args, parameters)`
|`add_meta_details`|Copy a meta_dict and add new parameters to it| Obvious bro
|`Plotter`|A class that handles plot sizing and can save the dataframe used to generate a plot as well as the plot itself|

## Getting Started
To get started, use this template to create a new repo then look at the following folders / files in order:


 1. [configs](configs): Houses various .yaml files. You can add any yaml files here and they will be read in to the parameters dictionary and can be made easily available in any part of the code. The [private_vars.yaml](configs/private_vars.yaml) file is meant to hold all parameters that could identify you. This makes it easier to ensure your repository is anonymized. Update it less frequently. 
 2. [setup](setup): Contains set up instructions, make sure to add the Python version you are using to the [README](setup/README.md). The [short requirements](setup/short_requirements.txt) are meant to store just the installed packages with appropriate versions, while the [long requirements](setup/long_requirements.txt) are meant to be generated with `pip freeze >> setup/long_requirements.txt`
 3. The [main](main.py) file contains the pattern that should typically be your entry point for most if not all calls to the codebase. Any command you want to call should be implemented as a click command and added as a subcommand to the main group in a file. In the click decorator of the `main` function of the calling file, you can specify any other config file parameters that you would like to be able to easily override in a command line call. Make sure to set the default appropriately. You may call this script in the following way: `python main.py --random_seed 38 --any_prespecified_config_file_overides subcommand_name --subcommand_specific_argument idk`

If you want to run any of these through a bash script, it may be helpful to use this template for argparse with bash. To enable loading of config variables into the bash scripts, see how to create a configs/config.sh file in the [README](configs/README.md)
```bash
#!/usr/bin/env bash

source configs/config.env || { echo "configs/config.env not found"; exit 1; }

# Define defaults and required args
declare -A ARGS
ARGS["first"]="hello"
ARGS["second"]="100000"
REQUIRED_ARGS=("third")

# Handle parsing and input errors below:
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

# Add Required to string
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

# Add Optionals to string
for opt in "${!ARGS[@]}"; do
    # Only list if NOT in required (to avoid double listing)
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

# 3. Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            # Extract the name (remove the leading --)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi            
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# 4. Validation
for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

# Print active variables
echo "Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Put your script code below:
```
