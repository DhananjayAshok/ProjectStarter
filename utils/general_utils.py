import os


def file_makedir(file_path):
    dir = os.path.dirname(file_path)
    if dir != "" and not os.path.exists(dir):
        os.makedirs(dir)
    return
