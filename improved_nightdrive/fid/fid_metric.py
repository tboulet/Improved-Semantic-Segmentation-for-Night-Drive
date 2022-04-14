#File for computing FID (distance metric between real night images distribution and generated night images distribution)


import math
import sys
import numpy as np
import tensorflow as tf
from scipy import linalg
from tqdm import tqdm
import pathlib

BATCH_SIZE = 14
N_IMAGES = 28
IMAGE_SIZE = (256, 256)
path_night_test_images = "ForkGAN/datasets/BDD100K/testB"    #Path to night images (real) 
log_filename = "ForkGAN/datasets/BDD100K/fid_logs"

print("Loading Inception model...")
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
    for img in tqdm(images):
        embeddings = inception_model.predict(img)
        image_embeddings.extend(embeddings)
    return np.array(image_embeddings)


#ORIGINAL NIGHT IMAGE DISTRIBUTION
print("Loading original night images...")
night_original_images = load_image(path_night_test_images)
real_embeddings = load_embedding(night_original_images)
mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
del night_original_images
del real_embeddings



#FID
def fid(path_night_generated_images, save_score = True):  
    '''Compute FID between night images and generated (day 2 night) night images. Possibly save its value into a file.
    path_night_generated_images : the path to generated images'''
    print("Computing FID...")  
    
    #GENERATED NIGHT IMAGE DISTRIBUTION
    night_generated_images = load_image(path_night_generated_images)
    generated_embeddings = load_embedding(night_generated_images)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    del night_generated_images
    del generated_embeddings
    
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
    
    #Save score into file data/fid_logs
    if save_score:
        
        try:
            file = open(log_filename, 'a')
        except FileNotFoundError:
            file = open(log_filename, 'w')
        file.write(f"{fid} for night generated images '{pathlib.PurePath(path_night_generated_images)}'")
        file.close()
    
    return fid
    

def get_list_of_fids():
    fid_list = list()
    for line in open(log_filename, 'r'):
        fid = float(line.split()[0])
        fid_list.append(fid)
    return fid_list




if __name__ == '__main__':
    print("Score FID:", fid(path_night_generated_images=sys.argv[1])
