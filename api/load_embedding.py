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