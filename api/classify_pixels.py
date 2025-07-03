# api/load_embedding.py
from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from geotessera import GeoTessera
import base64
import io
from PIL import Image

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse query parameters
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            
            lat = float(params.get('lat', [0])[0])
            lon = float(params.get('lon', [0])[0])
            year = int(params.get('year', [2024])[0])
            
            # Initialize GeoTessera client
            tessera = GeoTessera(version="v1")
            
            # Load embedding data
            embedding = tessera.get_embedding(lat, lon, year)
            
            # Convert to serializable format
            # Note: We'll compress the data significantly for web transfer
            height, width, channels = embedding.shape
            
            # For web transfer, we'll send a downsampled version and statistics
            # Full resolution processing will happen on subsequent API calls
            downsample_factor = 4  # Reduce by 4x for web transfer
            small_height = height // downsample_factor
            small_width = width // downsample_factor
            
            # Downsample embedding for web display
            downsampled = np.zeros((small_height, small_width, channels), dtype=np.float32)
            for i in range(small_height):
                for j in range(small_width):
                    # Average over downsample window
                    y_start = i * downsample_factor
                    y_end = min((i + 1) * downsample_factor, height)
                    x_start = j * downsample_factor
                    x_end = min((j + 1) * downsample_factor, width)
                    
                    downsampled[i, j] = np.mean(
                        embedding[y_start:y_end, x_start:x_end], 
                        axis=(0, 1)
                    )
            
            # Convert to list for JSON serialization
            embedding_data = downsampled.tolist()
            
            # Calculate statistics for each channel
            channel_stats = []
            for c in range(channels):
                channel_data = embedding[:, :, c]
                stats = {
                    'min': float(np.min(channel_data)),
                    'max': float(np.max(channel_data)),
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data))
                }
                channel_stats.append(stats)
            
            response = {
                'success': True,
                'data': {
                    'embedding': embedding_data,
                    'original_shape': [height, width, channels],
                    'downsampled_shape': [small_height, small_width, channels],
                    'downsample_factor': downsample_factor,
                    'location': {'lat': lat, 'lon': lon},
                    'year': year,
                    'channel_stats': channel_stats
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

---

# api/visualize_bands.py
from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from geotessera import GeoTessera
import base64
import io
from PIL import Image

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            lat = data['lat']
            lon = data['lon']
            year = data.get('year', 2024)
            bands = data['bands']  # [r_band, g_band, b_band]
            normalize = data.get('normalize', True)
            
            # Initialize GeoTessera client
            tessera = GeoTessera(version="v1")
            
            # Load full resolution embedding
            embedding = tessera.get_embedding(lat, lon, year)
            height, width, channels = embedding.shape
            
            # Extract specified bands
            vis_data = embedding[:, :, bands].copy()
            
            # Normalize if requested
            if normalize:
                for i in range(3):
                    channel = vis_data[:, :, i]
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:
                        vis_data[:, :, i] = (channel - min_val) / (max_val - min_val)
            
            # Convert to 8-bit RGB
            vis_data = np.clip(vis_data * 255, 0, 255).astype(np.uint8)
            
            # Create PIL Image
            img = Image.fromarray(vis_data)
            
            # Convert to base64 for web transfer
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            response = {
                'success': True,
                'data': {
                    'image': f'data:image/png;base64,{img_base64}',
                    'shape': [height, width],
                    'bands': bands,
                    'normalized': normalize
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

---

# api/classify_pixels.py
from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from geotessera import GeoTessera
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import base64
import io
from PIL import Image

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            lat = data['lat']
            lon = data['lon']
            year = data.get('year', 2024)
            training_samples = data['training_samples']  # List of {x, y, class_id}
            method = data.get('method', 'knn')  # 'knn' or 'kmeans'
            threshold = data.get('threshold', 0.5)
            
            # Initialize GeoTessera client
            tessera = GeoTessera(version="v1")
            
            # Load full resolution embedding
            embedding = tessera.get_embedding(lat, lon, year)
            height, width, channels = embedding.shape
            
            # Reshape embedding for classification
            embedding_flat = embedding.reshape(-1, channels)
            
            if method == 'knn' and training_samples:
                # Extract training data
                train_embeddings = []
                train_labels = []
                
                for sample in training_samples:
                    x, y, class_id = sample['x'], sample['y'], sample['class_id']
                    if 0 <= x < width and 0 <= y < height:
                        embed_idx = y * width + x
                        train_embeddings.append(embedding_flat[embed_idx])
                        train_labels.append(class_id)
                
                if len(train_embeddings) > 0:
                    train_embeddings = np.array(train_embeddings)
                    train_labels = np.array(train_labels)
                    
                    # Use KNN for classification
                    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
                    knn.fit(train_embeddings)
                    
                    # Classify all pixels
                    distances, indices = knn.kneighbors(embedding_flat)
                    
                    # Apply threshold
                    predictions = np.full(len(embedding_flat), -1)  # -1 = unclassified
                    mask = distances.flatten() < threshold
                    predictions[mask] = train_labels[indices.flatten()[mask]]
                    
                    # Reshape back to image dimensions
                    classification_result = predictions.reshape(height, width)
                else:
                    classification_result = np.full((height, width), -1)
                    
            elif method == 'kmeans':
                # Unsupervised clustering
                n_clusters = data.get('n_clusters', 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embedding_flat)
                classification_result = labels.reshape(height, width)
                
            else:
                raise ValueError("Invalid method or no training samples provided")
            
            # Calculate classification statistics
            unique_classes, counts = np.unique(classification_result, return_counts=True)
            class_stats = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
            
            # Create visualization image
            vis_img = self.create_classification_image(classification_result, data.get('class_colors', {}))
            
            # Convert to base64
            buffer = io.BytesIO()
            vis_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            response = {
                'success': True,
                'data': {
                    'classification': classification_result.tolist(),
                    'visualization': f'data:image/png;base64,{img_base64}',
                    'class_stats': class_stats,
                    'shape': [height, width],
                    'method': method,
                    'threshold': threshold if method == 'knn' else None
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def create_classification_image(self, classification_result, class_colors):
        """Create a colored visualization of classification results."""
        height, width = classification_result.shape
        
        # Default colors for classes
        default_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Light Blue
            (255, 128, 128) # Pink
        ]
        
        # Create RGB image
        rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        unique_classes = np.unique(classification_result)
        for i, class_id in enumerate(unique_classes):
            if class_id == -1:  # Unclassified
                continue
                
            mask = classification_result == class_id
            
            # Get color for this class
            if str(class_id) in class_colors:
                color = class_colors[str(class_id)]
                # Convert hex to RGB if needed
                if isinstance(color, str) and color.startswith('#'):
                    color = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
            else:
                color = default_colors[i % len(default_colors)]
            
            rgb_img[mask] = color
        
        return Image.fromarray(rgb_img)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
