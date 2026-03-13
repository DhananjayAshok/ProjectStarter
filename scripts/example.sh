#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Define defaults and required args. 
# These should be specific to this script and not shared across scripts (that is handled below).
declare -A ARGS
ARGS["batch_size"]="32"
ARGS["lr"]="0.001"
# there are two ways of handling none values:
ARGS["scheduler"]="none" 
ARGS["train_only"]=""
# keep in mind, you must only use "" if you are not going to be using the pattern --flag value. 
# keep in mind also, if you use "none", then any python scripts taking this flag in will need to know to process it to a None. 
REQUIRED_ARGS=("third")

# extend it to include the shared arguments (if any) (place the function in utils)
populate_common_optional_training_args ARGS # don't pass in $, we want to pass the name of the variable for it to modify
populate_common_required_training_args REQUIRED_ARGS # same here. Make sure to do this before allowed flags is computed. 

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
echo validating: "${REQUIRED_ARGS[@]}"
for req in "${REQUIRED_ARGS[@]}"; do
    echo $req : "${ARGS[$req]}"
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
arg_string=$(args_to_flags ARGS)
echo $arg_string
exp_name=$(python scripts/get_strings.py exp_name $arg_string)
echo "Do something with exp: $exp_name"