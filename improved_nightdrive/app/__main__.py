from typing import List

import gradio as gr
import numpy as np
import tensorflow as tf

from improved_nightdrive.pipeline.lighting import (
    apply_gamma_map,
    apply_threshold,
)
from improved_nightdrive.pipeline.pipeline import full_prediction
from improved_nightdrive.pipeline.preprocess import GammaProcess, GaussianBlur
from improved_nightdrive.segmentation.models import make_model

tf.get_logger().setLevel("ERROR")


def lighting_gradio(
    input: np.ndarray,
    lighting: float,
    class_to_light: str,
    blur_list: List[bool],
    model_name: str,
):
    model_type = model_name.split(" ")[-1]
    config = {
        "model_name": model_type,
        "image_size": 224,
        "intermediate_size": (225, 400),
        "num_classes": 19,
        "new_classes": 5,
    }
    class_to_int = {
        "Route": 0,
        "Obstacles": 1,
        "Panneaux": 2,
        "Usagers fragiles": 3,
        "Usagers": 4,
    }
    blur_bool = {
        "Blur before": False,
        "Blur after": False,
    }
    for blur in blur_list:
        blur_bool[blur] = True

    model = make_model(config)
    if model_name == "Nuit unetmobilenetv2":
        model.load_weights("./results/unet_night_gamma_75_75_25.h5")
        process = [GammaProcess(p=[0.75, 0.75, 0.25])]
    elif model_name == "Nuit deeplabv3":
        model.load_weights("./results/deeplab_night_gamma_75_25_50.h5")
        process = [GammaProcess(p=[0.75, 0.25, 0.5])]
    elif model_name == "Jour&Nuit unetmobilenetv2":
        model.load_weights("./results/weighted_unet_both_none.h5")
        process = []
    elif model_name == "ForkGAN unetmobilenetv2":
        model.load_weights("./results/unet_forkgan.h5")
        process = []
    else:
        raise ValueError(f"{model_name} is unknown !")

    prediction, input = full_prediction(input, config, model, process)
    if blur_bool["Blur before"]:
        prediction = GaussianBlur(5, 5).blur(prediction)
    prediction = apply_threshold(prediction, 0.5)
    if blur_bool["Blur after"]:
        if class_to_light == "Usagers fragiles":
            prediction = GaussianBlur(10, 5).blur(prediction)
        elif class_to_light == "Panneaux":
            prediction = GaussianBlur(5, 1).blur(prediction)
        else:
            prediction = GaussianBlur(25, 20).blur(prediction)

    gamma = 1 - lighting
    modified_input = apply_gamma_map(
        input, prediction, class_to_int[class_to_light], max_gamma=1, min_gamma=gamma
    )

    return modified_input.numpy()


def launch_app():
    demo = gr.Interface(
        lighting_gradio,
        [
            gr.Image(type="numpy"),
            gr.Slider(0.0, 1.0),
            gr.Radio(["Route", "Obstacles", "Panneaux", "Usagers fragiles", "Usagers"]),
            gr.CheckboxGroup(["Blur before", "Blur after"]),
            gr.Radio(
                [
                    "Nuit deeplabv3",
                    "Nuit unetmobilenetv2",
                    "Jour&Nuit unetmobilenetv2",
                    "ForkGAN unetmobilenetv2",
                ]
            ),
        ],
        [
            gr.Image(type="numpy"),
        ],
    )
    demo.launch()


if __name__ == "__main__":
    launch_app()
