# api/get_pixel_embedding.py
from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from geotessera import GeoTessera

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
            pixel_x = data['x']
            pixel_y = data['y']
            
            # Initialize GeoTessera client
            tessera = GeoTessera(version="v1")
            
            # Load embedding
            embedding = tessera.get_embedding(lat, lon, year)
            height, width, channels = embedding.shape
            
            # Validate pixel coordinates
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                raise ValueError(f"Pixel coordinates ({pixel_x}, {pixel_y}) out of bounds for image size ({width}, {height})")
            
            # Extract pixel embedding
            pixel_embedding = embedding[pixel_y, pixel_x, :].tolist()
            
            # Calculate some statistics
            embedding_stats = {
                'min': float(np.min(pixel_embedding)),
                'max': float(np.max(pixel_embedding)),
                'mean': float(np.mean(pixel_embedding)),
                'std': float(np.std(pixel_embedding))
            }
            
            response = {
                'success': True,
                'data': {
                    'embedding': pixel_embedding,
                    'coordinates': {'x': pixel_x, 'y': pixel_y},
                    'stats': embedding_stats,
                    'channels': channels
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