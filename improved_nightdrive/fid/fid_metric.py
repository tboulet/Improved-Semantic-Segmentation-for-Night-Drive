"""FID Metric"""
from typing import Tuple

import numpy as np
from scipy import linalg
import tensorflow as tf
from tqdm import tqdm


from improved_nightdrive.segmentation.models import (
    DeeplabV3,
    get_encoder_deeplabv3,
)


class FID:
    def __init__(
        self,
        encoder: str = "inception",
        batch_size: int = 8,
        image_size: Tuple[int, int] = (512, 256),
        path_night_real_images: str = "ForkGAN/datasets/BDD100K/trainB",  # Path to night images (real)
        log_filename: str = "ForkGAN/datasets/BDD100K/fid_logs",
    ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.image_size = image_size

        self.path_night_real_images = path_night_real_images
        self.log_filename = log_filename

        if self.encoder == "inception":
            print("Loading Inception model...")
            self.net = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", pooling="avg"
            )

        elif self.encoder == "deeplabv3":
            print("Loading Deeplabv3 model...")
            model = DeeplabV3(224, 19)
            model.load_weights(self.encoder)
            self.net = get_encoder_deeplabv3(model)

        else:
            raise ValueError(f"{self.encoder} is not a valid encoder name")

        self.original_mu, self.original_sigma = self.compute_original_distribution()

    def load_images(self, path: str) -> tf.data.Dataset:
        """Loads images from directory"""
        images = tf.keras.utils.image_dataset_from_directory(
            path,
            label_mode=None,
            seed=420,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )
        return images

    def pre_process_images(self, images: tf.Tensor) -> tf.data.Dataset:
        """Preprocesses images for encoder"""
        if self.encoder == "inception":
            images = tf.keras.applications.inception_v3.preprocess_input(images)
        else:
            images = tf.image.resize(images, (448, 224))
            images = images / 255.0
            images = tf.image.random_crop(images, (images.shape[0], 224, 224, 3))
        return images

    def load_embeddings(self, images: tf.data.Dataset) -> np.ndarray:
        """Loads the embeddings given an encoder"""
        image_embeddings = []
        for batch_img in tqdm(images, desc="Loading embeddings... "):
            batch_img = self.pre_process_images(batch_img)
            embeddings = self.net.predict(batch_img)
            image_embeddings.extend(embeddings)
        return np.array(image_embeddings)

    # ORIGINAL NIGHT IMAGE DISTRIBUTION
    def compute_real_distribution(self) -> Tuple[float, float]:
        """Computes the real distribution"""
        print("Loading real night images...")
        night_original_images = self.load_images(self.path_night_real_images)
        real_embeddings = self.load_embeddings(night_original_images)
        mu, sigma = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        return mu, sigma

    # FID
    def __call__(self, path_night_generated_images: str) -> float:
        """Compute FID score between real and generated night images

        Args:
            path_night_generated_images (str)
        Returns:
            fid_score (float)
        """
        print("Computing FID...")

        # GENERATED NIGHT IMAGE DISTRIBUTION
        night_generated_images = self.load_images(path_night_generated_images)
        generated_embeddings = self.load_embeddings(night_generated_images)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(
            generated_embeddings, rowvar=False
        )
        del night_generated_images
        del generated_embeddings

        # COMPUTE FID
        # calculate sum squared difference between means
        ssdiff = np.sum((self.original_mu - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(np.dot(self.original_sigma, sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            # calculate score
        fid = ssdiff + np.trace(self.original_sigma + sigma2 - 2.0 * covmean)

        return fid
