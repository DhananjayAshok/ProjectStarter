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
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
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

### Handling optional args that may be absent: use `"none"`, not `""`

Optional ARGS must always have a non-empty default (the blank-check above enforces this). For args that are genuinely optional at the Python level, use `"none"` as the default:

```bash
ARGS["scheduler"]="none"
```

`args_to_flags` will emit `--scheduler none`, so any Python script receiving it must convert the string `"none"` to `None`. If you don't want to touch the Python side, handle it in bash before building `arg_string`:

```bash
# Do this in the script body below "Put your script code below:", not in the setup section
if [[ "${ARGS["scheduler"]}" == "none" ]]; then
    scheduler_arg=""
else
    scheduler_arg="--scheduler ${ARGS["scheduler"]}"
fi
```

Then pass `$scheduler_arg` directly rather than including `scheduler` in `args_to_flags`.

---

When multiple scripts share common args, define `populate_*` functions in `scripts/utils.sh`. Call them after your script-specific `ARGS` and `REQUIRED_ARGS`, but before `ALLOWED_FLAGS`:

```bash
populate_common_optional_training_args ARGS          # pass name, not $ARGS
populate_common_required_training_args REQUIRED_ARGS

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
```

This merges the shared defaults/required keys into your script's arrays in place.

### Adding a new shared optional arg

In `scripts/utils.sh`, add a key-value pair to the defaults dict:

```bash
declare -A COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS=(["num_epochs"]="10" ["new_arg"]="default_val")
```

That's it. Every script calling `populate_common_optional_training_args` will automatically receive `new_arg`.

### Adding a new shared required arg

In `scripts/utils.sh`, add to the required array:

```bash
COMMON_REQUIRED_TRAINING_ARGS=("dataset" "model" "new_required_arg")
```

Every script calling `populate_common_required_training_args` will now require `--new_required_arg` at the command line.

### Adding a new group of shared args

If you have a new set of scripts that share different args from the training group, follow the same pattern in `utils.sh`:

```bash
declare -A COMMON_OPTIONAL_EVAL_ARGS_DEFAULTS=(["split"]="test")
function populate_common_optional_eval_args() {
    populate_dict COMMON_OPTIONAL_EVAL_ARGS_DEFAULTS "$1"
}

COMMON_REQUIRED_EVAL_ARGS=("checkpoint")
function populate_common_required_eval_args() {
    populate_array COMMON_REQUIRED_EVAL_ARGS "$1"
}

COMMON_EVAL_ARGS_KEYS=("${COMMON_REQUIRED_EVAL_ARGS[@]}" "${!COMMON_OPTIONAL_EVAL_ARGS_DEFAULTS[@]}")
```

Then call `populate_common_optional_eval_args ARGS` / `populate_common_required_eval_args REQUIRED_ARGS` in your eval scripts, and pass `COMMON_EVAL_ARGS_KEYS` (no `$`, no `[@]}`) to `args_to_flags_subset` when calling eval subscripts.

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

Usage (when you need the raw flag string for other purposes):

```bash
arg_string=$(args_to_flags ARGS)
```

For calling `get_strings.py`, prefer `get_string_from_args` which handles this internally.

Note: bash associative arrays have no guaranteed iteration order, so flag order may vary. This is fine for `--key value` style CLI args, and the scripts/get_string.py file handles this well.

---

## `args_to_flags_subset`: Calling a subscript without passing all ARGS

When script B calls script A, passing `args_to_flags ARGS` would include B's script-specific flags, which A would reject as unknown. Use `args_to_flags_subset` to emit only the keys that A expects.

`utils.sh` pre-computes a combined key list from the required args array and the optional defaults dict keys. Pass that single list:

```bash
subset=$(args_to_flags_subset ARGS COMMON_TRAINING_ARGS_KEYS)
bash scripts/a.sh $subset
```

`COMMON_TRAINING_ARGS_KEYS` is defined once in `utils.sh` after both sources exist:

```bash
COMMON_TRAINING_ARGS_KEYS=("${COMMON_REQUIRED_TRAINING_ARGS[@]}" "${!COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS[@]}")
```

When adding new shared args, update only the relevant source (`COMMON_REQUIRED_TRAINING_ARGS` or `COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS`) — `COMMON_TRAINING_ARGS_KEYS` picks them up automatically.

---

## `scripts/get_strings.py`: Python string generation from bash

Sometimes bash needs a computed string (e.g. an experiment name) that is easier to build in Python. `scripts/get_strings.py` provides a registry of named `StringFunction` classes callable from bash.

### Usage: `get_string_from_args`

The preferred way to call it is via the `get_string_from_args` helper in `utils.sh`, which handles the `args_to_flags` conversion internally:

```bash
exp_name=$(get_string_from_args exp_name ARGS)
if [[ -z "$exp_name" ]]; then
    echo "Error: failed to build experiment name"
    exit 1
fi
model_save_path="$storage_dir/models/$exp_name/"
```

Pass the string function name and the ARGS variable name (no `$`). All `none` values in ARGS are automatically skipped by the parser, so you can pass ARGS directly without pre-filtering.

### How it works

1. Each `StringFunction` subclass declares `NAME`, `REQUIRED_ARGS`, and `OPTIONAL_ARGS`.
2. `get_string_from_args` serializes ARGS to `--key value` flags and calls `scripts/get_strings.py`.
3. It prints exactly one string to stdout (captured by bash) and exits non-zero on error.

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
- `none` values are **automatically skipped** by the parser — a `--key none` flag is never passed to `_get_string`. This means you can safely pass the full `$arg_string` (which may contain `none` values) without those keys showing up as unexpected args or overriding defaults in `OPTIONAL_ARGS`


---

## Example Script

See the working example [below](scripts/example.sh) to understand how these pieces come together. 

