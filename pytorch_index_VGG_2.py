import os
import h5py
import numpy as np
from pytorch_VGG_1 import VGGNet

# Define the directory containing the images
images_path = "./fashion-dataset/fashion-dataset/images"
img_list = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')]

print("Start feature extraction")

# Initialize the VGGNet model
model = VGGNet()

# Prepare lists to store features and image names
feats = []
names = []

# Extract features for each image
for im in img_list:
    print(f"Extracting features from image: {im}")
    X = model.extract_feat(im)

    feats.append(X)
    names.append(im)

feats = np.array(feats)

# Directory for storing extracted features
output = "VGG16Features_pytorch.h5"
print("Writing feature extraction results to h5 file")

# Write the features and image names to an HDF5 file
h5f = h5py.File(output, 'w')
h5f.create_dataset('dataset_1', data=feats)
h5f.create_dataset('dataset_2', data=np.string_(names))
h5f.close()

