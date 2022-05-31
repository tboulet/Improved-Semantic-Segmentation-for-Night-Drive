from absl import app, flags
import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

from improved_nightdrive.segmentation.analyse_dataset import class_proportion

flags.DEFINE_string(
    "dataset_y_path",
    "/media/raffaelbdl/T7/BDD100K/bdd100k/day/labels/",
    "Dataset images",
)
FLAGS = flags.FLAGS


def get_class_proportion(dataset_y_path: str):
    """Returns proportion for each class"""
    proportions = class_proportion(dataset_y_path, True)
    prop_str = ""
    for (p, n) in zip(
        proportions, ["Route", "Obstacles", "Panneaux", "Usagers fragiles", "Usagers"]
    ):
        prop_str += n
        prop_str += " "
        prop_str += f"{p:.2f} | "
    print("\n", prop_str, "\n")
    return proportions


def main(_):
    get_class_proportion(FLAGS.dataset_y_path)


if __name__ == "__main__":
    app.run(main)
