'''
path_night_test_images: path to folder containing test images for night
path_day_test_images: path to folder containing test images for day
'''

import math
import sys
import numpy as np
import tensorflow as tf
from scipy import linalg
from tqdm import tqdm


BATCH_SIZE = 14
N_IMAGES = 28
IMAGE_SIZE = (256, 256)
path_night_test_images = "dataset/testB"
path_night_generated_images = "dataset/trainB"  #Not trainB (real night images) but some generated images

print("Loading Inception model...")
count = math.ceil(N_IMAGES/BATCH_SIZE)
inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg')

def load_image(path):
    images = tf.keras.utils.image_dataset_from_directory(
    path,
    label_mode = None,
    seed=420,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)
    return images
    
def load_embedding(images):
    # return inception_model.predict(images)
    image_embeddings = []
    for _ in range(count):
        img = next(iter(images))
        embeddings = inception_model.predict(img)
        image_embeddings.extend(embeddings)
    return np.array(image_embeddings)


#ORIGINAL NIGHT IMAGE DISTRIBUTION
print("Loading original night images...")
night_original_images = load_image(path_night_test_images)
real_embeddings = load_embedding(night_original_images)
mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)



#FID
def fid(path_night_generated_images):  
    print("Computing FID...")  
    
    #GENERATED NIGHT IMAGE DISTRIBUTION
    night_generated_images = load_image(path_night_generated_images)
    generated_embeddings = load_embedding(night_generated_images)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    
    #COMPUTE FID
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    



if __name__ == '__main__':
    print("Score FID:", fid(path_night_generated_images=path_night_generated_images))