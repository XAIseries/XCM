import os
import shutil
from distutils.dir_util import copy_tree


def create_logger(log_filename, display=True):
    """Create a log file for the experiment"""
    f = open(log_filename, "a")
    counter = [0]

    def logger(text):
        if display:
            print(text)
        f.write(text + "\n")
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close


def makedir(path):
    """If the directory does not exist, create it"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_experiment(xp_dir, configuration):
    """Save files about the experiment"""
    makedir(xp_dir)
    makedir(xp_dir + "grad-cam")
    shutil.copy(src="./main.py", dst=xp_dir)
    shutil.copy(src=configuration, dst=xp_dir)
    copy_tree(src="./models/", dst=xp_dir + "/models/")
