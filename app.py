import os
import sys
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from fuzzywuzzy import fuzz

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
df = None
DEPENDENCIES_LOADED = False

def load_dependencies():
    """Load dependencies with error handling"""
    global DEPENDENCIES_LOADED
    try:
        import pandas as pd
        import numpy as np
        from math import radians, sin, cos, sqrt, atan2
        from fuzzywuzzy import fuzz
        DEPENDENCIES_LOADED = True
        logger.info("✓ All dependencies loaded successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Error importing dependencies: {e}")
        DEPENDENCIES_LOADED = False
        return False

def load_hospital_data():
    """Try to load hospital data from multiple formats"""
    if not DEPENDENCIES_LOADED:
        logger.error("Dependencies not loaded, skipping data loading")
        return None

    current_dir = os.getcwd()
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    data_files = [
        ("hospital_data.csv", "csv"),
        ("hospital_data.json", "json"),
        ("hospitals.csv", "csv"),
        ("data.csv", "csv"),
        ("hospital_data.pkl", "pickle")
    ]
    
    for filename, file_type in data_files:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            logger.info(f"Found {file_type} file: {filename}")
            try:
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                    if 'Specialties' in df.columns:
                        df['Specialties'] = df['Specialties'].apply(safe_eval_specialties)
                    logger.info(f"✓ Successfully loaded {len(df)} records from CSV")
                    return df
                elif file_type == "json":
                    df = pd.read_json(file_path)
                    logger.info(f"✓ Successfully loaded {len(df)} records from JSON")
                    return df
                elif file_type == "pickle":
                    import pickle
                    with open(file_path, 'rb') as f:
                        df = pickle.load(f)
                    logger.info(f"✓ Successfully loaded {len(df)} records from pickle")
                    return df
            except Exception as e:
                logger.error(f"✗ Error loading {filename}: {str(e)}")
                continue
    
    logger.error("No valid data file found")
    return None

def safe_eval_specialties(x):
    """Safely evaluate specialties column"""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    if isinstance(x, str):
        try:
            if x.startswith('[') and x.endswith(']'):
                result = eval(x)
                return result if isinstance(result, list) else [str(result)]
            else:
                return [item.strip() for item in x.split(',') if item.strip()]
        except:
            return [x] if x else []
    elif isinstance(x, list):
        return x
    else:
        return [str(x)] if x else []

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers"""
    if not DEPENDENCIES_LOADED:
        return 0
    
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def recommend_hospital(user_lat, user_lon, disease, df, top_n=5, similarity_threshold=80):
    """Recommend nearest hospitals for a specific disease"""
    if not DEPENDENCIES_LOADED:
        return {"error": "Dependencies not loaded"}
    
    if df is None or df.empty:
        return {"error": "No valid hospital data available"}
    
    # Validate user coordinates
    if not (-90 <= user_lat <= 90 and -180 <= user_lon <= 180):
        return {"error": "Invalid coordinates: latitude must be -90 to 90, longitude -180 to 180"}

    # Clean and standardize
    df = df.copy()
    disease = disease.lower().strip()

    # Handle different specialty column formats
    specialty_columns = ['Specialties', 'specialties', 'Specialty', 'specialty']
    specialty_col = None
    for col in specialty_columns:
        if col in df.columns:
            specialty_col = col
            break
    
    if specialty_col is None:
        logger.warning("No specialty column found, using all hospitals")
        matching_hospitals = df.copy()
    else:
        def match_disease(specialties):
            if pd.isna(specialties) or not specialties:
                return False
            if isinstance(specialties, str):
                return fuzz.partial_ratio(disease, specialties.lower()) >= similarity_threshold
            elif isinstance(specialties, list):
                for specialty in specialties:
                    if fuzz.partial_ratio(disease, str(specialty).lower()) >= similarity_threshold:
                        return True
            return False

        matching_hospitals = df[df[specialty_col].apply(match_disease)].copy()

        if matching_hospitals.empty:
            logger.warning(f"No hospitals found for '{disease}', using all hospitals")
            matching_hospitals = df.copy()

    if matching_hospitals.empty:
        return {"error": "No hospitals found"}

    # Calculate distances
    matching_hospitals['Distance'] = matching_hospitals.apply(
        lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1
    )

    # Sort by distance and select top N
    nearest_hospitals = matching_hospitals.sort_values(by='Distance').head(top_n)

    # Prepare result with available columns
    result_columns = [
        'Hospital_Name', 'Address_Original_First_Line', 'State', 'District', 
        'Pincode', 'Telephone', 'Mobile_Number', 'Emergency_Num', 'Facilities', 'Distance'
    ]
    available_columns = [col for col in result_columns if col in nearest_hospitals.columns]
    
    result = nearest_hospitals[available_columns].to_dict(orient='records')
    return result

# Initialize everything
logger.info("Initializing Hospital Recommendation API...")
load_dependencies()
df = load_hospital_data()

# Health check endpoint
@app.route('/health')
def health():
    """Health check endpoint"""
    current_dir = os.getcwd()
    files_in_dir = os.listdir('.')
    
    debug_info = {
        "status": "healthy",
        "dependencies_loaded": DEPENDENCIES_LOADED,
        "data_loaded": df is not None,
        "hospital_count": len(df) if df is not None else 0,
        "current_directory": current_dir,
        "files_in_directory": files_in_dir[:10],
        "python_version": sys.version
    }
    
    if df is not None:
        debug_info["sample_columns"] = list(df.columns)
        
    return jsonify(debug_info)

# API endpoint for Android app
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """Main API endpoint for hospital recommendations"""
    try:
        if not DEPENDENCIES_LOADED:
            return jsonify({"error": "Server dependencies not loaded"}), 500
        
        data = request.get_json()
        logger.info(f"Received POST request data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required parameters
        if 'latitude' not in data or data['latitude'] is None:
            return jsonify({"error": "Missing or null latitude parameter"}), 400
        if 'longitude' not in data or data['longitude'] is None:
            return jsonify({"error": "Missing or null longitude parameter"}), 400
        if 'disease' not in data or not data['disease'].strip():
            return jsonify({"error": "Missing or empty disease parameter"}), 400
        
        # Convert coordinates
        try:
            user_lat = float(data['latitude'])
            user_lon = float(data['longitude'])
        except (ValueError, TypeError) as e:
            logger.error(f"Coordinate conversion error: {str(e)}, latitude: {data.get('latitude')}, longitude: {data.get('longitude')}")
            return jsonify({"error": f"Invalid numeric coordinates: {str(e)}"}), 400
        
        disease = data.get('disease', '').strip()
        try:
            top_n = int(data.get('top_n', 5))
            if top_n <= 0:
                return jsonify({"error": "top_n must be a positive integer"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid top_n: must be a positive integer"}), 400
        
        # Get recommendations
        recommendations = recommend_hospital(user_lat, user_lon, disease, df, top_n)
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            return jsonify(recommendations), 400
        
        return jsonify({
            "success": True,
            "count": len(recommendations),
            "hospitals": recommendations,
            "search_params": {
                "latitude": user_lat,
                "longitude": user_lon,
                "disease": disease,
                "top_n": top_n
            }
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in POST /api/recommend: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# GET endpoint for testing
@app.route('/api/recommend', methods=['GET'])
def api_recommend_get():
    """GET endpoint for testing"""
    try:
        if not DEPENDENCIES_LOADED:
            return jsonify({"error": "Server dependencies not loaded"}), 500
            
        logger.info(f"Received GET request args: {request.args}")
        
        # Validate required parameters
        if 'latitude' not in request.args or request.args['latitude'] is None:
            return jsonify({"error": "Missing or null latitude parameter"}), 400
        if 'longitude' not in request.args or request.args['longitude'] is None:
            return jsonify({"error": "Missing or null longitude parameter"}), 400
        if 'disease' not in request.args or not request.args['disease'].strip():
            return jsonify({"error": "Missing or empty disease parameter"}), 400
        
        # Convert coordinates
        try:
            user_lat = float(request.args['latitude'])
            user_lon = float(request.args['longitude'])
        except (ValueError, TypeError) as e:
            logger.error(f"Coordinate conversion error: {str(e)}, latitude: {request.args.get('latitude')}, longitude: {request.args.get('longitude')}")
            return jsonify({"error": f"Invalid numeric coordinates: {str(e)}"}), 400
            
        disease = request.args.get('disease', '').strip()
        try:
            top_n = int(request.args.get('top_n', 5))
            if top_n <= 0:
                return jsonify({"error": "top_n must be a positive integer"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid top_n: must be a positive integer"}), 400
        
        # Get recommendations
        recommendations = recommend_hospital(user_lat, user_lon, disease, df, top_n)
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            return jsonify(recommendations), 400
        
        return jsonify({
            "success": True,
            "count": len(recommendations),
            "hospitals": recommendations,
            "search_params": {
                "latitude": user_lat,
                "longitude": user_lon,
                "disease": disease,
                "top_n": top_n
            }
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in GET /api/recommend: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Route for the home page
@app.route('/')
def home():
    """Home page with API documentation"""
    status = "✓ Ready" if DEPENDENCIES_LOADED and df is not None else "✗ Not Ready"
    data_count = len(df) if df is not None else 0
    
    return f"""
    <h1>🏥 Hospital Recommendation API</h1>
    <p><strong>Status:</strong> {status}</p>
    <p><strong>Hospital Records:</strong> {data_count}</p>
    
    <h3>📍 API Endpoints:</h3>
    <ul>
        <li><strong>POST /api/recommend</strong> - Main API endpoint</li>
        <li><strong>GET /api/recommend</strong> - Testing endpoint</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    
    <h3>🔧 Usage Example:</h3>
    <p><strong>POST /api/recommend</strong> with JSON:</p>
    <pre>{{
    "latitude": 19.9975,
    "longitude": 73.7898,
    "disease": "cardiology",
    "top_n": 5
}}</pre>
    
    <p><strong>GET Test:</strong></p>
    <a href="/api/recommend?latitude=19.9975&longitude=73.7898&disease=cardiology&top_n=3">
        /api/recommend?latitude=19.9975&longitude=73.7898&disease=cardiology&top_n=3
    </a>
    """

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)