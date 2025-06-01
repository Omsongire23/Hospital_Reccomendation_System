from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from fuzzywuzzy import fuzz
import pickle
import os

app = Flask(__name__)

# Try multiple possible pickle file names
POSSIBLE_PICKLE_NAMES = [
    "hospital_data.pkl",
    "hospitals.pkl", 
    "data.pkl",
    "hospital_dataset.pkl",
    "hospital_info.pkl"
]

def find_and_load_pickle():
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Try different paths
    possible_paths = [
        "hospital_data.pkl",
        os.path.join(current_dir, "hospital_data.pkl"),
        "./hospital_data.pkl"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found pickle file at: {path}")
            try:
                with open(path, 'rb') as f:
                    df = pickle.load(f)
                print(f"Successfully loaded {len(df)} hospital records")
                return df
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue
    
    print("No pickle file found with any of the expected paths")
    return None

def load_pickle_data(pickle_path):
    try:
        # Check if file exists first
        if not os.path.exists(pickle_path):
            print(f"File {pickle_path} does not exist")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            return None
            
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        print(f"Successfully loaded {len(df)} hospital records")
        return df
    except FileNotFoundError:
        print(f"Error: Pickle file {pickle_path} not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")
        return None

# Load DataFrame
df = find_and_load_pickle()

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

    # Clean and standardize specialties (should already be clean from pickle)
    df = df.copy()  # Create a copy to avoid modifying the original
    disease = disease.lower().strip()

    # Fuzzy matching for specialties
    def match_disease(specialties):
        if not specialties:
            return False
        for specialty in specialties:
            if fuzz.partial_ratio(disease, str(specialty).lower()) >= similarity_threshold:
                return True
        return False

    # Filter hospitals with matching specialties
    matching_hospitals = df[df['Specialties'].apply(match_disease)].copy()

    # If no hospitals match, fall back to hospitals with empty specialties
    if matching_hospitals.empty:
        matching_hospitals = df[df['Specialties'].apply(len) == 0].copy()

    if matching_hospitals.empty:
        return {"error": "No hospitals found for the specified disease."}

    # Calculate distances to all matching hospitals
    matching_hospitals.loc[:, 'Distance'] = matching_hospitals.apply(
        lambda row: haversine(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1
    )

    # Sort by distance and select top N
    nearest_hospitals = matching_hospitals.sort_values(by='Distance').head(top_n)

    # Prepare the result
    result = nearest_hospitals[[
        'Hospital_Name', 'Address_Original_First_Line', 'State', 'District', 'Pincode',
        'Telephone', 'Mobile_Number', 'Emergency_Num', 'Facilities', 'Distance'
    ]].to_dict(orient='records')

    return result

# Health check endpoint
@app.route('/health')
def health():
    current_dir = os.getcwd()
    files_in_dir = os.listdir('.')
    pickle_exists = os.path.exists("hospital_data.pkl")
    
    debug_info = {
        "status": "healthy",
        "data_loaded": df is not None,
        "hospital_count": len(df) if df is not None else 0,
        "current_directory": current_dir,
        "files_in_directory": files_in_dir,
        "pickle_file_exists": pickle_exists
    }
    
    if df is not None:
        debug_info["sample_columns"] = list(df.columns)
        
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

# Route for the home page with the input form (optional - for testing)
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

# Route to handle the recommendation request (keeping for backward compatibility)
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        user_lat = float(request.form['latitude'])
        user_lon = float(request.form['longitude'])
        disease = request.form['disease']

        # Get recommendations
        recommendations = recommend_hospital(user_lat, user_lon, disease, df)

        # Return JSON response instead of template
        return jsonify({
            "success": True,
            "hospitals": recommendations
        })
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter valid numeric coordinates."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)