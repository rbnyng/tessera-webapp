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