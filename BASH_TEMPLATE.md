# Bash Templates

Reference for bash scripting patterns used in this project family.

---

## Standalone Script Template

Every bash script must source `configs/utils.sh` (provides `storage_dir`, `WANDB_PROJECT`, and all other YAML params as shell variables and activate the virtual environment). 

```bash
#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

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

---

## Shared Bash Infrastructure (`scripts/utils.sh`)

When multiple scripts share logic — most commonly constructing experiment or environment names — define reusable functions in `scripts/utils.sh`. This file handles sourcing so downstream scripts only need one line.

---

## Value-Returning Functions in `utils.sh`

Bash functions return values by printing to stdout, which the caller captures with `$(...)`. This means any diagnostic `echo` inside the function **must** be redirected to stderr with `>&2`, otherwise it gets captured as part of the return value.

### Example: experiment name builder

```bash
# In scripts/utils.sh

function get_exp_name() {
    declare -A ARGS
    REQUIRED_ARGS=("algorithm" "timesteps" "gamma" "env")

    ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}")

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --*)
                FLAG=${1#--}
                VALID=false
                for allowed in "${ALLOWED_FLAGS[@]}"; do
                    if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
                done
                if [ "$VALID" = false ]; then
                    echo "Error: Unknown flag --$FLAG" >&2  # >&2 so it is not captured
                    return 1
                fi
                ARGS["$FLAG"]="$2"
                shift 2
                ;;
            *) echo "Unknown argument: $1" >&2; return 1 ;;
        esac
    done

    for req in "${REQUIRED_ARGS[@]}"; do
        if [[ -z "${ARGS[$req]}" ]]; then
            echo "Error: missing required argument --$req" >&2
            FAILED=true
        fi
    done
    if [ "$FAILED" = true ]; then return 1; fi

    # Diagnostic messages go to stderr — visible in terminal but not captured by $(...)
    echo "Building experiment name..." >&2

    # The return value is the only thing printed to stdout
    echo "${ARGS["algorithm"]}-${ARGS["timesteps"]}-${ARGS["gamma"]}-${ARGS["env"]}"
}
```

### Calling a value-returning function

```bash
# Capture the return value with $(...)
exp_name=$(get_exp_name --algorithm sac --timesteps 1000000 --gamma 0.99 --env default)

# Always check for failure
if [[ -z "$exp_name" ]]; then
    echo "Error: failed to build experiment name"
    exit 1
fi

echo "Running experiment: $exp_name"
model_save_path="$storage_dir/models/$exp_name/"
```

The `>&2` diagnostic lines print to the terminal as normal. Only the final `echo` (the actual name string) is captured into `$exp_name`.

### Key rules for value-returning functions
- Use `return 1` (not `exit 1`) on error — `exit` would kill the calling shell
- All `echo` statements except the return value must use `>&2`
- The caller must check that the result is non-empty before using it
