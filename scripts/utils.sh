source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }

# args_to_flags <assoc_array_name>
#
# Converts a bash associative array into a flat --key value string suitable
# for passing to a Python click command or another bash script.
# Empty values ("") are emitted as --key none.
#
# Usage (capture-safe — all diagnostics go to stderr):
#   declare -A ARGS=( ["lr"]="0.001" ["dataset"]="" )
#   flags=$(args_to_flags ARGS)
#   python get_strings.py <string_kind> $flags
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then
            val="none"
        fi
        result+="--${key} ${val} "
    done
    echo "${result% }"  # trim trailing space
}


function populate_common_optional_training_args() {
    local -n _dict="$1"
    _dict["num_epochs"]="10"
}

function populate_common_required_training_args() {
    local -n _arr="$1"
    _arr+=("dataset" "model")
}