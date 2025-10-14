from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_info, log_warn
import click
from huggingface_hub import HfApi, Repository
import os

loaded_parameters = load_parameters()



@click.command()
@click.option("--repo_namespace", type=str, default=None, required=False, help="The namespace (user or org) on the hub where the repo will be created.")
@click.option("--repo_name", type=str, required=True, help="The name of the repo to create on the hub.")
@click.option("--private", is_flag=True, default=False, help="Whether the repo should be private.")
@click.pass_obj
def create_hub_repo(parameters, repo_namespace, repo_name, private):
    """
    Create a new repo on the Hugging Face Hub.
    """
    if repo_namespace is None:
        repo_id = repo_name
    else:
        repo_id = f"{repo_namespace}/{repo_name}"
    api = HfApi()
    api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True, # Won't raise an error if the repo already exists
    private=private)


@click.command()
@click.option("--repo_namespace", type=str, default=None, required=False, help="The namespace (user or org) on the hub where the repo is located.")
@click.option("--repo_name", type=str, required=True, help="The name of the repo on the hub.")
@click.pass_obj
def setup_sync(parameters, repo_namespace, repo_name):
    """
    Set up the local sync directory to sync with the specified repo on the Hugging Face Hub.
    """
    repo_id = repo_name if repo_namespace is None else f"{repo_namespace}/{repo_name}"
    sync_dir = os.path.abspath(parameters["sync_dir"])
    if not os.path.exists(sync_dir):
        log_error(f"Sync directory {sync_dir} does not exist, please check your configuration.", parameters)
    repo = Repository(
        local_dir=sync_dir,
        clone_from=repo_id,
        repo_type="dataset",
    )
    log_info(f"Successfully set up sync with repo {repo_namespace}/{repo_name} in directory {sync_dir}.", parameters)
    repo.pull()


@click.command()
@click.pass_obj
def pull_data_from_hub(parameters):
    """
    Pull the latest data from the Hugging Face Hub into the local sync directory.
    """
    sync_dir = os.path.abspath(parameters["sync_dir"])
    if not os.path.exists(sync_dir):
        log_error(f"Sync directory {sync_dir} does not exist, please check your configuration.", parameters)
    repo = Repository(local_dir=sync_dir)
    #TODO: Error out if repo is not set up
    repo.pull()
    log_info(f"Successfully pulled latest data from the hub into {sync_dir}.", parameters)


@click.command()
@click.option("--local_path", type=str, required=True, help="The local path within the root directory to push to the hub. This must always be a local path relative to the storage_dir indicated in the config file.")
@click.option("--ignore_patterns", type=str, multiple=True, default=[], help="Patterns to ignore when pushing data to the hub.")
@click.pass_obj
def push_data_to_hub(parameters, local_path, ignore_patterns):
    """
    In general how this works is:
        1. Check if repo exists. If it does, first pull and write it to sync directory (i.e. root_dir)
    """
    root_dir = parameters["storage_dir"]
    sync_dir = parameters["sync_dir"]
    full_path = os.path.join(root_dir, local_path)
    if not os.path.exists(full_path):
        log_error(f"Local path {full_path} does not exist, please check your configuration.", parameters)
    # check that full_path is not inside sync_dir, to avoid recursive issues
    if os.path.abspath(sync_dir) in os.path.abspath(full_path):
        log_error(f"Local path {full_path} cannot be inside sync_dir {sync_dir}, please check your configuration.", parameters)
    


@click.group()
@click.pass_context
def main(ctx, **input_parameters):
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters


main.add_command(create_hub_repo, name="create")
main.add_command(setup_sync, name="init")
main.add_command(pull_data_from_hub, name="pull")
main.add_command(push_data_to_hub, name="push")

if __name__ == "__main__":
    main()