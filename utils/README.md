# Utils Description

The utils file offers some important functionality across the project:


| Function / Function Set| Description |  Common Pattern
| - | - |  -
| `load_parameters` | Used to read the parameters from the config files and return a populated dictionary. You can also pass in an existing parameters dictionary, in which case this dict will just be returned | `parameters = load_parameters(parameters)`
| `log_error`, `log_warn`, `log_info`, `log_dict` | Used to send specific messages to the logger. Takes an optional parameter argument that matters if and only if you pass in a `--log_file` argument | `log_warn('Bruh')` if you do not intend on passing in a log_file, `log_warn('Bruh', parameters)` if you do
|`write_meta` |Used to save a dictionary of arguments to a specified path. This hashes the dict based on its argument values, allowing you to create a unique identifier for a particular configuration of a run. Should be used when you create an artifact (e.g. a model) that has hyperparameters you want to keep track of| `args={'random_seed': 42}` and then `write_meta('data_process/', args, parameters)`
|`add_meta_details`|Copy a meta_dict and add new parameters to it| Obvious bro
|`Plotter`|A class that handles plot sizing and can save the dataframe used to generate a plot as well as the plot itself|
