from flask import Flask, request, jsonify, render_template
import os
import pickle
import json

# Initialize Flask app first
app = Flask(__name__)

# Try to import pandas and numpy with error handling
try:
    import pandas as pd
    import numpy as np
    from math import radians, sin, cos, sqrt, atan2
    from fuzzywuzzy import fuzz
    DEPENDENCIES_LOADED = True
    print("All dependencies loaded successfully")
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    DEPENDENCIES_LOADED = False

def load_hospital_data():
    """Try to load hospital data from multiple formats"""
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Try different file formats in order of preference
    data_files = [
        ("hospital_data.csv", "csv"),
        ("hospital_data.json", "json"),
        ("hospital_data.pkl", "pickle"),
        ("hospitals.csv", "csv"),
        ("hospitals.json", "json"),
        ("hospitals.pkl", "pickle"),
        ("data.csv", "csv"),
        ("data.json", "json"),
        ("data.pkl", "pickle")
    ]
    
    for filename, file_type in data_files:
        if os.path.exists(filename):
            print(f"Found {file_type} file: {filename}")
            try:
                if file_type == "csv":
                    df = pd.read_csv(filename)
                    print(f"Successfully loaded {len(df)} records from CSV")
                    return df
                elif file_type == "json":
                    df = pd.read_json(filename)
                    print(f"Successfully loaded {len(df)} records from JSON")
                    return df
                elif file_type == "pickle":
                    # Try multiple pickle loading methods
                    df = load_pickle_safe(filename)
                    if df is not None:
                        return df
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
    
    print("No valid data file found")
    return None

def load_pickle_safe(filepath):
    """Safely load pickle file with multiple methods"""
    methods = [
        lambda: pickle.load(open(filepath, 'rb')),
        lambda: pd.read_pickle(filepath),
        lambda: pickle.load(open(filepath, 'rb'), encoding='latin1'),
        lambda: pickle.load(open(filepath, 'rb'), fix_imports=True),
    ]
    
    for i, method in enumerate(methods):
        try:
            print(f"Trying pickle loading method {i+1}")
            df = method()
            print(f"Successfully loaded {len(df)} records using method {i+1}")
            return df
        except Exception as e:
            print(f"Method {i+1} failed: {str(e)}")
            continue
    
    return None

# Load DataFrame
df = load_hospital_data()

# Haversine formula to calculate distance between two coordinates (in kilometers)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to recommend the nearest hospital for a specific disease
def recommend_hospital(user_lat, user_lon, disease, df, top_n=1, similarity_threshold=80):
    if df is None or df.empty:
        return {"error": "No valid hospital data available."}
    
    # Validate user coordinates
    if not (-90 <= user_lat <= 90 and -180 <= user_lon <= 180):
        return {"error": "Invalid coordinates provided. Latitude must be between -90 and 90, Longitude between -180 and 180."}

    # Clean and standardize specialties
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
        print("Warning: No specialty column found, using all hospitals")
        matching_hospitals = df.copy()
    else:
        # Fuzzy matching for specialties
        def match_disease(specialties):
            if pd.isna(specialties) or not specialties:
                return False
            # Handle both string and list formats
            if isinstance(specialties, str):
                return fuzz.partial_ratio(disease, specialties.lower()) >= similarity_threshold
            elif isinstance(specialties, list):
                for specialty in specialties:
                    if fuzz.partial_ratio(disease, str(specialty).lower()) >= similarity_threshold:
                        return True
            return False

        # Filter hospitals with matching specialties
        matching_hospitals = df[df[specialty_col].apply(match_disease)].copy()

        # If no hospitals match, use all hospitals
        if matching_hospitals.empty:
            print(f"No hospitals found for '{disease}', using all hospitals")
            matching_hospitals = df.copy()

    if matching_hospitals.empty:
        return {"error": "No hospitals found."}

    # Calculate distances to all matching hospitals
    matching_hospitals.loc[:, 'Distance'] = matching_hospitals.apply(
        lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1
    )

    # Sort by distance and select top N
    nearest_hospitals = matching_hospitals.sort_values(by='Distance').head(top_n)

    # Prepare the result with flexible column names
    result_columns = [
        'Hospital_Name', 'Address_Original_First_Line', 'State', 'District', 'Pincode',
        'Telephone', 'Mobile_Number', 'Emergency_Num', 'Facilities', 'Distance'
    ]
    
    # Use available columns
    available_columns = [col for col in result_columns if col in nearest_hospitals.columns]
    
    result = nearest_hospitals[available_columns].to_dict(orient='records')
    return result

# Health check endpoint
@app.route('/health')
def health():
    current_dir = os.getcwd()
    files_in_dir = os.listdir('.')
    
    debug_info = {
        "status": "healthy",
        "dependencies_loaded": DEPENDENCIES_LOADED,
        "data_loaded": df is not None,
        "hospital_count": len(df) if df is not None else 0,
        "current_directory": current_dir,
        "files_in_directory": files_in_dir
    }
    
    if df is not None:
        debug_info["sample_columns"] = list(df.columns)
        debug_info["sample_data"] = df.head(1).to_dict(orient='records')
        
    return jsonify(debug_info)

# API endpoint for Android app
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        user_lat = float(data.get('latitude'))
        user_lon = float(data.get('longitude'))
        disease = data.get('disease', '').strip()
        top_n = int(data.get('top_n', 5))  # Default to 5 hospitals
        
        if not disease:
            return jsonify({"error": "Disease parameter is required"}), 400
        
        # Get recommendations
        recommendations = recommend_hospital(user_lat, user_lon, disease, df, top_n)
        
        # Check if there's an error in recommendations
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
        
    except ValueError as e:
        return jsonify({"error": "Invalid input. Please provide valid numeric coordinates."}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing required parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# GET endpoint for testing
@app.route('/api/recommend', methods=['GET'])
def api_recommend_get():
    try:
        # Get query parameters
        user_lat = float(request.args.get('latitude'))
        user_lon = float(request.args.get('longitude'))
        disease = request.args.get('disease', '').strip()
        top_n = int(request.args.get('top_n', 5))
        
        if not disease:
            return jsonify({"error": "Disease parameter is required"}), 400
        
        # Get recommendations
        recommendations = recommend_hospital(user_lat, user_lon, disease, df, top_n)
        
        # Check if there's an error in recommendations
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
        
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input. Please provide valid numeric coordinates."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Route for the home page
@app.route('/')
def home():
    return """
    <h1>Hospital Recommendation API</h1>
    <p>API Endpoints:</p>
    <ul>
        <li><strong>POST /api/recommend</strong> - Main API endpoint for Android app</li>
        <li><strong>GET /api/recommend</strong> - Testing endpoint with query parameters</li>
        <li><strong>GET /health</strong> - Health check endpoint</li>
    </ul>
    <h3>Usage Example:</h3>
    <p>POST /api/recommend with JSON body:</p>
    <pre>
    {
        "latitude": 19.9975,
        "longitude": 73.7898,
        "disease": "cardiology",
        "top_n": 5
    }
    </pre>
    <p>GET /api/recommend?latitude=19.9975&longitude=73.7898&disease=cardiology&top_n=5</p>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)