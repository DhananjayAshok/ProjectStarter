This project uses [uv](https://docs.astral.sh/uv/guides/projects/) to manage dependencies. One notable difference is that while the project setup with uv is usually done at the root level, we will set up the environment at the [setup](../setup) level instead. 

First, ensure you have installed uv in your python. This project used python 3.XX
```console
pip install --upgrade pip  uv
```


# Replicating Existing UV Project

Make sure there is a `uv.lock` and `pyproject.toml` file in the setup directory. Then, run:

```console
uv sync
```

This will create a virtual environment in setup/.venv. Before running any code in this repo, make sure to run (from root):

```console
source setup/.venv/bin/activate
```


# Creating New UV Project

If you want to create a new project, first cd into the [setup](../setup) directory and then run:
```console
uv init
rm main.py
```
We never use this main file, so delete it. This will create a project with the name `setup`. To change the project name, edit the `pyproject.toml` file. 

You can now add pip packages with:
```console
uv add click
```

This creates a uv.lock file, that must be committed to version control
