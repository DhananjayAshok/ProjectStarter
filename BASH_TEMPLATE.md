# Bash Templates

Reference for bash scripting patterns used in this project family.

---

## Quick Reference

| Task | Function |
|------|----------|
| Serialize ARGS dict to `--key value` flags | `args_to_flags ARGS` |
| Serialize only a subset of keys | `args_to_flags_subset ARGS KEYS_ARRAY` |
| Get a computed string (e.g. exp name) from Python | `get_string_from_args <name> ARGS` |
| Merge shared optional defaults into ARGS | `populate_common_optional_training_args ARGS` |
| Merge shared required keys into REQUIRED_ARGS | `populate_common_required_training_args REQUIRED_ARGS` |
| Copy all keys from one dict to another | `populate_dict SRC DST` |
| Append all elements from one array to another | `populate_array SRC DST` |
| Copy a subset of keys between dicts | `populate_dict_subset SRC DST KEYS` |
| Copy a subset of elements between arrays | `populate_array_subset SRC DST KEYS` |

All functions use bash namerefs — pass variable **names**, not values (no `$`).

---

## Script Boilerplate

Every script follows this structure. See `scripts/example.sh` for a working example.

```bash
#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["batch_size"]="32"
ARGS["scheduler"]="none"   # use "none" for absent optionals, never ""

REQUIRED_ARGS=("dataset")

# OPTIONAL: merge shared args from utils.sh (do this BEFORE ALLOWED_FLAGS)
populate_common_optional_training_args ARGS
populate_common_required_training_args REQUIRED_ARGS

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Put your script code below:
```

### Optional args: always handle `"none"` as `None`

The blank-check above enforces that all optional defaults are non-empty. For args that are genuinely absent at the Python level, use `"none"` as the default. `args_to_flags` will emit `--key none`, and `get_strings.py` will pass `None` to `_get_string`.

If you don't want Python to receive it at all, handle it in bash:
```bash
if [[ "${ARGS["scheduler"]}" == "none" ]]; then
    scheduler_arg=""
else
    scheduler_arg="--scheduler ${ARGS["scheduler"]}"
fi
# Then pass $scheduler_arg directly, not via args_to_flags
```

---

## Shared Args Pattern

When multiple scripts share common args, define them once in `utils.sh` and merge them in at the top of each script.

### Defining a shared group (in `utils.sh`)

```bash
declare -A COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS=(["num_epochs"]="10" ["lr"]="0.001")
function populate_common_optional_training_args() {
    populate_dict COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS "$1"
}

COMMON_REQUIRED_TRAINING_ARGS=("dataset" "model")
function populate_common_required_training_args() {
    populate_array COMMON_REQUIRED_TRAINING_ARGS "$1"
}

# Pre-computed key list for args_to_flags_subset calls
COMMON_TRAINING_ARGS_KEYS=("${COMMON_REQUIRED_TRAINING_ARGS[@]}" "${!COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS[@]}")
```

Adding a new optional arg: add to `COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS`. Adding a new required arg: add to `COMMON_REQUIRED_TRAINING_ARGS`. `COMMON_TRAINING_ARGS_KEYS` picks up both automatically.

### Using in a script

```bash
populate_common_optional_training_args ARGS          # pass name, not $ARGS
populate_common_required_training_args REQUIRED_ARGS # call before ALLOWED_FLAGS
```

### Inheriting a subset of another group's defaults

When a new group shares some defaults with an existing one, use `populate_dict_subset` to avoid duplicating values:

```bash
SAME_AS_TRAINING=("lr" "batch_size")
declare -A MY_GROUP_DEFAULTS
populate_dict_subset COMMON_OPTIONAL_TRAINING_ARGS_DEFAULTS MY_GROUP_DEFAULTS SAME_AS_TRAINING
MY_GROUP_DEFAULTS["my_extra_arg"]="default_val"
```

---

## `args_to_flags` and `args_to_flags_subset`

`args_to_flags` serializes a bash associative array into `--key value` form. Empty values become `none`.

```bash
flags=$(args_to_flags ARGS)
python scripts/some_script.py $flags
```

When script B calls script A, use `args_to_flags_subset` to emit only the keys A accepts:

```bash
subset=$(args_to_flags_subset ARGS COMMON_TRAINING_ARGS_KEYS)
bash scripts/a.sh $subset
```

---

## `scripts/get_strings.py`: Python String Generation

For computed strings (experiment names, paths) that are easier to build in Python, use the `get_strings.py` registry via `get_string_from_args`:

```bash
exp_name=$(get_string_from_args exp_name ARGS)
if [[ -z "$exp_name" ]]; then echo "Error: failed to build exp name"; exit 1; fi
model_save_path="$storage_dir/models/$exp_name/"
```

`none` values in ARGS are passed as `None` to the Python function.

### Adding a new string function (in `scripts/get_strings.py`)

```python
class MyExpName(StringFunction):
    NAME = "my_exp_name"
    REQUIRED_ARGS = ["dataset", "model"]
    OPTIONAL_ARGS = {"version": "v1"}   # prefer controlling defaults via utils.sh instead

    def _get_string(self, **kwargs) -> str:
        return f"{kwargs['dataset']}_{kwargs['model']}_{kwargs['version']}"

STRING_FUNCTIONS = [..., MyExpName]
```

Rules:
- `_get_string` must return a string with **no spaces** if used as a path
- Unexpected arguments are silently ignored — keep `REQUIRED_ARGS` minimal
- `none` values arrive as `None` in kwargs; handle or ignore as needed

---

## Example Script

See [scripts/example.sh](scripts/example.sh) for a complete working example of all the above.
