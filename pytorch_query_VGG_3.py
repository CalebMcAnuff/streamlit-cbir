from pytorch_VGG_1 import VGGNet
import numpy as np
import h5py
from scipy.spatial import distance
import pandas as pd

# Load the indexed features and image names from the HDF5 file
h5f = h5py.File("VGG16Features_pytorch.h5", 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# Define the query image
queryImg = "./fashion-dataset/fashion-dataset/images/46885.jpg"
print("Searching for similar images")

# Initialize the VGGNet model
model = VGGNet()

# Extract features from the query image
X = model.extract_feat(queryImg)

# Compute the cosine distance between the query image and all indexed images
scores = []
for feat in feats:
    score = 1 - distance.cosine(X, feat)
    scores.append(score)
scores = np.array(scores)

# Rank images by similarity score (highest to lowest)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

# Get top 3 matches
maxres = 3
imlist = [imgNames[index] for index in rank_ID[:maxres]]
print(f"Top {maxres} images in order are: {imlist}")
df_styles = pd.read_csv("./fashion-dataset/fashion-dataset/styles.csv")
for i in imlist:
    print([df_styles.id==i])

