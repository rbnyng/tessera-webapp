from vercel import app
from geotessera import GeoTessera
import json

tessera = GeoTessera(version="v1")

@app.route('/api/embeddings/<lat>/<lng>')
def get_embeddings(lat, lng):
    year = request.args.get('year', 2024)
    embedding = tessera.get_embedding(lat=float(lat), lon=float(lng), year=int(year))
    
    # Process and return embedding data
    return json.dumps(embedding.tolist())