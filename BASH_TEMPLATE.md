# Bash Templates

Reference for bash scripting patterns used in this project family.


## Shared Bash Infrastructure (`scripts/utils.sh`)

`scripts/utils.sh` handles sourcing `configs/config.env` and the virtualenv so downstream scripts only need one line. It also defines shared helper functions: `populate_common_*` functions for shared args, and `args_to_flags` for serializing ARGS to pass to Python (specifically for [getting strings](#scriptsget_stringspy-python-string-generation-from-bash)).

---

## Shared Parameters via `utils.sh` (nameref pattern)

All operation scripts must declare an ARGS array of required an optional parameters and read in the flags from the input. 

```bash
#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Define script-specific defaults and required args
declare -A ARGS
ARGS["first"]="hello"
ARGS["second"]="100000"
REQUIRED_ARGS=("third")

# OPTIONAL: Extend with shared arguments from utils.sh (call BEFORE ALLOWED_FLAGS is computed)
#populate_common_optional_training_args ARGS
#populate_common_required_training_args REQUIRED_ARGS

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

# Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
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

# Validation
for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Put your script code below:
```

When multiple scripts share a common set of optional defaults or required arguments, define `populate_*` functions in `scripts/utils.sh` using bash namerefs (`local -n`). This lets a function modify the caller's array in place without any return-value machinery.

```bash
# In scripts/utils.sh

function populate_common_optional_training_args() {
    local -n _dict="$1"
    _dict["num_epochs"]="10"
    # add more shared optional args here
}

function populate_common_required_training_args() {
    local -n _arr="$1"
    _arr+=("dataset" "model")
    # add more shared required args here
}
```

Call these **after** declaring your script-specific `ARGS` and `REQUIRED_ARGS`, but **before** computing `ALLOWED_FLAGS`:

```bash
declare -A ARGS
ARGS["lr"]="0.001"
REQUIRED_ARGS=("experiment_id")

populate_common_optional_training_args ARGS        # pass name, not $ARGS
populate_common_required_training_args REQUIRED_ARGS

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
```

Pass the **variable name as a plain string** (no `$`). Using `$ARGS` would expand the array contents instead of passing the name.

In this way, you don't need to repeat parameters that feature in multiple scripts. 

---

## `args_to_flags`: Converting ARGS to a CLI flag string

`args_to_flags` (defined in `scripts/utils.sh`) serializes a bash associative array into a `--key value` string suitable for passing to a Python script or another bash script. Empty values (`""`) are emitted as `none`.

```bash
# In scripts/utils.sh
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then val="none"; fi
        result+="--${key} ${val} "
    done
    echo "${result% }"
}
```

Usage:

```bash
arg_string=$(args_to_flags ARGS)
python scripts/get_strings.py exp_name $arg_string
```

Note: bash associative arrays have no guaranteed iteration order, so flag order may vary. This is fine for `--key value` style CLI args, and the scripts/get_string.py file handles this well. 

---

## `scripts/get_strings.py`: Python string generation from bash

Sometimes bash needs a computed string (e.g. an experiment name) that is easier to build in Python. `scripts/get_strings.py` provides a registry of named `StringFunction` classes callable from bash.

### How it works

1. Each `StringFunction` subclass declares `NAME`, `REQUIRED_ARGS`, and `OPTIONAL_ARGS`.
2. The script is called with the function name followed by `--key value` pairs.
3. It prints exactly one string to stdout (captured by bash) and exits non-zero on error.

```bash
exp_name=$(python scripts/get_strings.py exp_name $arg_string)
if [[ -z "$exp_name" ]]; then
    echo "Error: failed to build experiment name"
    exit 1
fi
```

### Adding a new string function

In `scripts/get_strings.py`:

```python
class MyExpName(StringFunction):
    NAME = "my_exp_name"
    REQUIRED_ARGS = ["dataset", "model"]
    OPTIONAL_ARGS = {"version": "v1"}

    def _get_string(self, **kwargs) -> str:
        return f"{kwargs['dataset']}_{kwargs['model']}_{kwargs['version']}"

STRING_FUNCTIONS = [..., MyExpName]
```

### Key rules
- `_get_string` must return a string with **no spaces** if it will be used as a path or experiment name
- Unexpected arguments passed in are silently ignored — keep required args minimal
- Optional args should have sensible defaults; prefer controlling defaults via `populate_common_optional_*` in `utils.sh` rather than hardcoding them here


---

## Example Script

See the working example [below](scripts/example.sh) to understand how these pieces come together. 

