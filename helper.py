import os


def mkdir_not_exists(path):
    if not os.path.exists(path):
        return os.makedirs(path)
    else:
        return True