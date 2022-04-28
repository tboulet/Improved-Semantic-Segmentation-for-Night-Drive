#File for computing FID (distance metric between real night images distribution and generated night images distribution)


import numpy as np
import tensorflow as tf
from scipy import linalg
from tqdm import tqdm
from improved_nightdrive.segmentation.models import UNet_MobileNetV2, get_encoder_unetmobilenetv2, DeeplabV3, get_encoder_deeplabv3


class FID:
    def __init__(
        self,
        encoder = "Inception",
        batch_size = 8,
        image_size = (512, 256),
        path_night_test_images = "ForkGAN/datasets/BDD100K/trainB",    #Path to night images (real)
        log_filename = "ForkGAN/datasets/BDD100K/fid_logs",
        ):
        self.name = "FID"
        self.ENCODER = encoder
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.path_night_test_images = path_night_test_images
        self.log_filename = log_filename

        if self.ENCODER == "Inception":
            print("Loading Inception model...")
            self.net = tf.keras.applications.InceptionV3(include_top=False, 
                                        weights="imagenet", 
                                        pooling='avg')

        else:
            print("Loading Segmentation model...")
            model = DeeplabV3(224, 19)
            model.load_weights(self.ENCODER)
            self.net = get_encoder_deeplabv3(model)
            self.net.summary()

        self.original_mu, self.original_sigma = self.compute_original_distribution()

    def load_image(self, path):
        images = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode = None,
        seed=420,
        image_size=self.IMAGE_SIZE,
        batch_size=self.BATCH_SIZE)
        return images

    def pre_process_image(self, imgs):
        if self.ENCODER == "Inception":
            imgs = tf.keras.applications.inception_v3.preprocess_input(imgs)
        else:
            imgs = tf.image.resize(imgs, (448, 224))
            imgs = imgs / 255
            imgs = tf.image.random_crop(imgs, (imgs.shape[0], 224, 224, 3))
        return imgs

    def load_embedding(self, images):
        image_embeddings = []
        for img in tqdm(images):
            img = self.pre_process_image(img)
            embeddings = tf.squeeze(self.net.predict(img))
            image_embeddings.extend(embeddings)
        return np.array(image_embeddings)


    #ORIGINAL NIGHT IMAGE DISTRIBUTION
    def compute_original_distribution(self):
        print("Loading original night images...")
        night_original_images = self.load_image(self.path_night_test_images)
        real_embeddings = self.load_embedding(night_original_images)
        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        return mu1, sigma1

    #FID
    def __call__(self, path_night_generated_images):  
        '''Compute FID between night images and generated (day 2 night) night images. Possibly save its value into a file.
        path_night_generated_images : the path to generated images'''
        print("Computing FID...")  

        #GENERATED NIGHT IMAGE DISTRIBUTION
        night_generated_images = self.load_image(path_night_generated_images)
        generated_embeddings = self.load_embedding(night_generated_images)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
        del night_generated_images
        del generated_embeddings
        
        #COMPUTE FID
        # calculate sum squared difference between means
        ssdiff = np.sum((self.original_mu - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(self.original_sigma.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            # calculate score
        fid = ssdiff + np.trace(self.original_sigma + sigma2 - 2.0 * covmean)

        return fid
