import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import cv2

# Load sample image (you can replace this with your own image)
image = cv2.imread("C:/Users/Lenovo/Downloads/cell 1.jpg")

# Convert image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to be a 2D array of pixels
image = np.array(image, dtype=np.float64) / 255
w, h, d = tuple(image.shape)
image_2d = np.reshape(image, (w * h, d))

# Shuffle the pixel intensities to randomize the order
image_2d_sample = shuffle(image_2d, random_state=0)[:1000]

# Perform K-means clustering
n_colors = 5 # Number of clusters (desired colors)
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(image_2d_sample)

# Get the labels assigned to each pixel
labels = kmeans.predict(image_2d)

# Assign different colors to different clusters
cluster_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],[1,0,1]])  # Specify different colors for each cluster as float RGB values

# Create a new image with different colors for each cluster
cluster_image = np.zeros_like(image)
for i in range(w):
    for j in range(h):
        pixel_label = labels[i * h + j]
        cluster_image[i, j, :] = cluster_colors[pixel_label]

# Display the original and cluster-colored images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')
ax2.imshow(cluster_image)
ax2.set_title(f"K-Means Cluster Image ({n_colors} colors)")
ax2.axis('off')
plt.show()
