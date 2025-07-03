import rasterio
from rasterio.transform import from_bounds
import numpy as np

# Your embedding data: shape (height, width, 1024)
embeddings = np.load('grid_0.15_52.05.npy')  

# Apply PCA to get 3 components
from sklearn.decomposition import PCA
height, width, channels = embeddings.shape
flat_embeddings = embeddings.reshape(-1, channels)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(flat_embeddings)
pca_embeddings = pca_result.reshape(height, width, 3)

# Normalize to 0-1 range
pca_normalized = (pca_embeddings - pca_embeddings.min()) / (pca_embeddings.max() - pca_embeddings.min())

# Define geographic bounds (example for San Francisco area)
west, south, east, north = 0.15, 52.5, 0.25, 52.0  # lon/lat bounds
transform = from_bounds(west, south, east, north, width, height)

# Save as GeoTIFF
with rasterio.open(
    'embeddings_pca.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=3,  # 3 bands
    dtype=rasterio.float32,
    crs='EPSG:4326',  # WGS84
    transform=transform,
    #compress='lzw'  # Optional compression
) as dst:
    # Write bands (rasterio expects bands-first: 3, height, width)
    dst.write(pca_normalized[:, :, 0], 1)  # PC1 -> Band 1
    dst.write(pca_normalized[:, :, 1], 2)  # PC2 -> Band 2  
    dst.write(pca_normalized[:, :, 2], 3)  # PC3 -> Band 3