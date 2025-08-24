from flask import Blueprint, jsonify, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from __init__ import db
from models import User, CropPrediction
import sqlalchemy
import requests
import json
import os
from datetime import datetime
import numpy as np
import time
import pandas as pd
import pickle

# Import ML model for crop prediction
try:
    from ml_model import get_ml_prediction, train_models_if_needed
    ML_MODEL_AVAILABLE = True
    print("Machine Learning crop prediction model loaded successfully!")
    # Train models if needed
    train_models_if_needed()
except ImportError as e:
    ML_MODEL_AVAILABLE = False
    print(f"Warning: ML model not available: {str(e)}. Using fallback prediction.")

# Import OpenAI model for pesticide/crop analysis
try:
    from model import (
        comprehensive_pesticide_analysis,
        analyze_pesticide_image,
        test_api_connection,
        simple_pesticide_analysis,
        analyze_crop_problem_with_ai
    )
    MODEL_AVAILABLE = True
    print("OpenAI pesticide analysis model loaded successfully!")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"Warning: OpenAI model.py not found or has import issues: {str(e)}. Using fallback analysis.")

main_bp = Blueprint('main', __name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "files", "crop_yield.csv")
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
WEATHER_API_KEY = 'f3922be0cb6dec52e9342d8829919685'
WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------- Crop labels ----------
crop_labels = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# ---------- Load dataset ----------
try:
    df = pd.read_csv(CSV_PATH)
    data_loaded = True
except FileNotFoundError:
    print("⚠️ Warning: Dataset not found. Dashboard disabled.")
    df = pd.DataFrame()
    data_loaded = False

# Template filters
@main_bp.app_template_filter('timestamp_to_time')
def timestamp_to_time(timestamp):
    """Convert UNIX timestamp to human-readable HH:MM:SS format."""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    except Exception:
        return "Invalid time"

@main_bp.app_template_filter('currency')
def currency_format(value):
    """Format currency for templates"""
    try:
        return f"₹{float(value):,.0f}"
    except (ValueError, TypeError):
        return "₹0"

# Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dashboard_data(crop=None):
    if not data_loaded:
        return {}
    
    filtered_df = df.copy()
    if crop and crop.lower() != 'all':
        filtered_df = filtered_df[filtered_df['Crop'].str.lower() == crop.lower()]
    
    # Remove "Whole Year" season
    if 'Season' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Season'] != 'Whole Year']
    
    # 1. Production Trend
    production_data = {'years': [], 'values': []}
    if 'Crop_Year' in filtered_df.columns and 'Production' in filtered_df.columns:
        prod_trend = filtered_df.groupby('Crop_Year')['Production'].sum()
        production_data = {
            'years': prod_trend.index.tolist(),
            'values': prod_trend.values.tolist()
        }
    
    # 2. Yield Trend
    yield_data = {'years': [], 'values': []}
    if 'Crop_Year' in filtered_df.columns and 'Yield' in filtered_df.columns:
        yield_trend = filtered_df.groupby('Crop_Year')['Yield'].mean()
        yield_data = {
            'years': yield_trend.index.tolist(),
            'values': yield_trend.values.tolist()
        }
    
    # 3. Season Distribution
    season_data = {'labels': [], 'values': []}
    if 'Season' in filtered_df.columns:
        season_dist = filtered_df['Season'].value_counts()
        season_data = {
            'labels': season_dist.index.tolist(),
            'values': season_dist.values.tolist()
        }
    
    # 4. Top States
    states_data = {'labels': [], 'values': []}
    if 'State' in filtered_df.columns and 'Production' in filtered_df.columns:
        top_states = filtered_df.groupby('State')['Production'].sum().nlargest(10)
        states_data = {
            'labels': top_states.index.tolist(),
            'values': top_states.values.tolist()
        }
    
    # 5. Average Yield by State
    avg_yield_state = {'labels': [], 'values': []}
    if 'State' in filtered_df.columns and 'Yield' in filtered_df.columns:
        top_states_yield = filtered_df.groupby('State')['Yield'].mean().nlargest(10)
        avg_yield_state = {
            'labels': top_states_yield.index.tolist(),
            'values': top_states_yield.values.tolist()
        }

    # 6. Scatter Plot (Area vs Yield by Season)
    scatter_data = []
    if all(col in filtered_df.columns for col in ['Season', 'Area', 'Yield']):
        for season in filtered_df['Season'].unique():
            season_points = filtered_df[filtered_df['Season'] == season]
            scatter_data.append({
                'name': season,
                'data': list(zip(season_points['Area'].tolist(), season_points['Yield'].tolist()))
            })
    
    # 7. Additional Chart
    if crop and crop.lower() != 'all':
        season_yield = filtered_df.groupby('Season')['Yield'].mean().sort_values()
        additional_chart = {
            'title': 'Yield by Season',
            'xlabel': 'Average Yield',
            'labels': season_yield.index.tolist(),
            'values': season_yield.values.tolist()
        }
    else:
        crop_div = filtered_df.groupby('State')['Crop'].nunique().nlargest(10)
        additional_chart = {
            'title': 'Top 10 States by Crop Diversity',
            'xlabel': 'Unique Crops',
            'labels': crop_div.index.tolist(),
            'values': crop_div.values.tolist()
        }

    return {
        'production_trend': production_data,
        'yield_trend': yield_data,
        'season_distribution': season_data,
        'top_states': states_data,
        'avg_yield_state': avg_yield_state,
        'scatter_data': scatter_data,
        'additional_chart': additional_chart
    }

def enhanced_crop_prediction(crop_type, season, area, soil_type, nitrogen, phosphorus, potassium, rainfall, temperature):
    """Enhanced crop prediction with multiple environmental factors"""
    
    crop_conditions = {
        'rice': {
            'temp_range': (20, 35),
            'rainfall_range': (1000, 2000),
            'soil_preference': ['clay', 'loam'],
            'nitrogen': (80, 120),
            'phosphorus': (40, 60),
            'potassium': (40, 60),
            'base_yield': 4.0,
            'season_preference': ['kharif']
        },
        'wheat': {
            'temp_range': (15, 25),
            'rainfall_range': (400, 1100),
            'soil_preference': ['loam', 'clay', 'silt'],
            'nitrogen': (100, 120),
            'phosphorus': (50, 60),
            'potassium': (40, 50),
            'base_yield': 3.5,
            'season_preference': ['rabi']
        },
        'maize': {
            'temp_range': (21, 30),
            'rainfall_range': (500, 1000),
            'soil_preference': ['loam', 'sandy'],
            'nitrogen': (120, 150),
            'phosphorus': (60, 80),
            'potassium': (60, 80),
            'base_yield': 5.2,
            'season_preference': ['kharif', 'rabi']
        },
        'cotton': {
            'temp_range': (21, 30),
            'rainfall_range': (500, 1000),
            'soil_preference': ['black', 'loam'],
            'nitrogen': (100, 150),
            'phosphorus': (50, 80),
            'potassium': (50, 80),
            'base_yield': 1.8,
            'season_preference': ['kharif']
        },
        'sugarcane': {
            'temp_range': (26, 32),
            'rainfall_range': (1000, 1500),
            'soil_preference': ['loam', 'clay'],
            'nitrogen': (200, 300),
            'phosphorus': (80, 120),
            'potassium': (100, 150),
            'base_yield': 70.0,
            'season_preference': ['perennial']
        }
    }
    
    # Get crop-specific conditions or use defaults
    conditions = crop_conditions.get(crop_type.lower(), {
        'temp_range': (20, 30),
        'rainfall_range': (600, 1200),
        'soil_preference': ['loam'],
        'nitrogen': (80, 120),
        'phosphorus': (40, 60),
        'potassium': (40, 60),
        'base_yield': 3.0,
        'season_preference': ['kharif', 'rabi']
    })
    
    # Calculate suitability scores
    temp_score = calculate_range_score(temperature, conditions['temp_range'])
    rainfall_score = calculate_range_score(rainfall, conditions['rainfall_range'])
    soil_score = 100 if soil_type.lower() in conditions['soil_preference'] else 70
    season_score = 100 if season.lower() in conditions['season_preference'] else 60
    
    # Nutrient adequacy scores
    n_score = calculate_range_score(nitrogen, conditions['nitrogen'])
    p_score = calculate_range_score(phosphorus, conditions['phosphorus'])
    k_score = calculate_range_score(potassium, conditions['potassium'])
    
    # Overall suitability score
    suitability_score = (temp_score + rainfall_score + soil_score + season_score + n_score + p_score + k_score) / 7
    
    # Calculate estimated yield
    base_yield = conditions['base_yield']
    yield_modifier = suitability_score / 100
    estimated_yield = base_yield * yield_modifier * area
    
    recommendations = []
    if temp_score < 80:
        if temperature < conditions['temp_range'][0]:
            recommendations.append("Consider protected cultivation or greenhouse farming due to low temperatures.")
        else:
            recommendations.append("Temperature is higher than optimal. Consider shade nets or cooling methods.")
    
    if rainfall_score < 80:
        if rainfall < conditions['rainfall_range'][0]:
            recommendations.append("Supplement with irrigation due to low rainfall.")
        else:
            recommendations.append("Ensure proper drainage due to excess rainfall.")
    
    if soil_score < 90:
        recommendations.append(f"Soil type is not optimal for {crop_type}. Consider soil amendments or choose a more suitable crop.")
    
    if season_score < 90:
        recommendations.append(f"Season is not optimal for {crop_type}. Consider planting in the recommended season.")
    
    # Nutrient recommendations
    if n_score < 80:
        recommendations.append(f"Increase nitrogen application. Recommended: {conditions['nitrogen'][0]}-{conditions['nitrogen'][1]} kg/ha")
    if p_score < 80:
        recommendations.append(f"Increase phosphorus application. Recommended: {conditions['phosphorus'][0]}-{conditions['phosphorus'][1]} kg/ha")
    if k_score < 80:
        recommendations.append(f"Increase potassium application. Recommended: {conditions['potassium'][0]}-{conditions['potassium'][1]} kg/ha")
    
    # Generate result message
    suitability_text = "Excellent" if suitability_score >= 90 else "Good" if suitability_score >= 75 else "Fair" if suitability_score >= 60 else "Poor"
    
    result = f"Based on your field conditions, {crop_type} cultivation shows {suitability_text.lower()} suitability ({suitability_score:.1f}%) for {season} season in {area} hectares."
    yield_estimation = f"Estimated yield: {estimated_yield:.2f} tons (Average: {estimated_yield/area:.2f} tons/hectare)"
    
    return {
        'result': result,
        'yield_estimation': yield_estimation,
        'suitability_score': round(suitability_score, 1),
        'recommendations': recommendations
    }

def calculate_range_score(value, optimal_range):
    """Calculate a score (0-100) based on how close a value is to the optimal range"""
    if value == 0:
        return 50  # Neutral score for missing data
    
    min_val, max_val = optimal_range
    
    if min_val <= value <= max_val:
        return 100  # Perfect score
    elif value < min_val:
        # Score decreases as value gets further from minimum
        deviation = (min_val - value) / min_val
        return max(0, 100 - (deviation * 100))
    else:
        # Score decreases as value gets further from maximum
        deviation = (value - max_val) / max_val
        return max(0, 100 - (deviation * 100))

def analyze_pesticide_with_ai(image_path, filename):
    """Analyze pesticide image using OpenAI model or fallback method."""
    try:
        if MODEL_AVAILABLE:
            print("Using OpenAI-based pesticide analysis...")
            
            # Check API connection first
            api_available, api_message = test_api_connection()
            
            if api_available:
                # Use the comprehensive OpenAI analysis
                result = comprehensive_pesticide_analysis(image_path, filename)
                
                # Ensure the result has the expected structure
                if not isinstance(result, dict):
                    raise ValueError("Invalid analysis result structure")
                
                # Add some additional metadata
                result['analysis_method'] = 'OpenAI GPT-4 Vision'
                result['api_status'] = 'success'
                
                print(f"OpenAI analysis completed successfully in {result.get('analysis_time', 'unknown')} seconds")
                return result
            else:
                print(f"OpenAI API not available: {api_message}. Using simple analysis.")
                return simple_pesticide_analysis(image_path, filename)
        else:
            print("OpenAI model not available. Using fallback analysis.")
            return get_fallback_analysis(filename)
            
    except Exception as e:
        print(f"Error in OpenAI analysis: {str(e)}")
        # Fall back to simple analysis if OpenAI fails
        try:
            if MODEL_AVAILABLE:
                return simple_pesticide_analysis(image_path, filename)
            else:
                return get_fallback_analysis(filename)
        except Exception as fallback_error:
            print(f"Fallback analysis also failed: {str(fallback_error)}")
            return get_emergency_fallback(filename)

def get_fallback_analysis(filename):
    """Enhanced fallback analysis when OpenAI model is not available."""
    # Try to extract information from filename
    filename_lower = filename.lower()
    
    # Determine pesticide type based on filename patterns
    if any(word in filename_lower for word in ['neem', 'organic', 'bio', 'natural']):
        return {
            'pesticide': 'Organic/Neem-based Pesticide',
            'confidence': 85,
            'safety': 'Safe',
            'recommendation': 'This appears to be an organic pesticide, likely neem-based. Safe for organic farming and beneficial insects when used properly. Apply during evening hours to minimize impact on pollinators. Effective against soft-bodied insects, aphids, and whiteflies. Can be used up to harvest time.',
            'active_ingredients': ['Azadirachtin', 'Neem oil extract'],
            'filename': filename,
            'analysis_method': 'Filename Pattern Analysis',
            'analysis_time': 0.05
        }
    elif any(word in filename_lower for word in ['roundup', 'glyphosate', 'herbicide']):
        return {
            'pesticide': 'Glyphosate-based Herbicide',
            'confidence': 88,
            'safety': 'Caution',
            'recommendation': 'This appears to be a glyphosate-based herbicide. Non-selective - will kill all vegetation. Use with caution around desired plants. Wear protective equipment including gloves and eye protection. Avoid application during windy conditions. Do not spray when rain is expected within 6 hours.',
            'active_ingredients': ['Glyphosate', 'Surfactants'],
            'filename': filename,
            'analysis_method': 'Filename Pattern Analysis',
            'analysis_time': 0.05
        }
    elif any(word in filename_lower for word in ['insecticide', 'pesticide', 'malathion', 'chlorpyrifos']):
        return {
            'pesticide': 'Chemical Insecticide',
            'confidence': 80,
            'safety': 'High Risk',
            'recommendation': 'This appears to be a chemical insecticide. High toxicity - use full protective equipment including respirator, gloves, and protective clothing. Follow strict safety protocols. Apply only when necessary and follow all label instructions carefully. Keep away from water sources and beneficial insects.',
            'active_ingredients': ['Various chemical compounds'],
            'filename': filename,
            'analysis_method': 'Filename Pattern Analysis',
            'analysis_time': 0.05
        }
    else:
        # Generic pesticide analysis
        return {
            'pesticide': 'General Purpose Pesticide',
            'confidence': 70,
            'safety': 'Caution',
            'recommendation': 'Unable to identify specific pesticide type. Follow all label instructions carefully. Use protective equipment including gloves and eye protection. Apply according to manufacturer guidelines. Store safely away from children and pets.',
            'active_ingredients': ['Refer to product label'],
            'filename': filename,
            'analysis_method': 'Generic Fallback Analysis',
            'analysis_time': 0.05
        }

def get_emergency_fallback(filename):
    """Emergency fallback when all other methods fail."""
    return {
        'pesticide': 'Unknown Pesticide Product',
        'confidence': 50,
        'safety': 'Caution',
        'recommendation': 'Unable to analyze the image automatically. Please consult the product label for specific instructions. Always follow safety guidelines: wear protective equipment, avoid windy conditions, and keep away from children and pets.',
        'active_ingredients': ['Consult product label'],
        'filename': filename,
        'analysis_method': 'Emergency Fallback',
        'analysis_time': 0.1
    }

# Routes
@main_bp.route('/')
def index():
    return render_template('index.html', model_loaded=ML_MODEL_AVAILABLE)

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    success_message = None
    error_message = None
    
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            subject = request.form.get('subject', 'General Inquiry')
            message = request.form.get('message')
            
            if not all([name, email, message]):
                error_message = "All fields are required."
            else:
                # Here you would integrate with your email service
                success_message = "Thank you for your message! We'll get back to you soon."
                flash(success_message, 'success')
                    
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            flash(error_message, 'error')
    
    return render_template('contact.html', success_message=success_message, error_message=error_message)

# Authentication routes
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@main_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        language = request.form.get('language', 'en')
        
        # Form validation
        if not all([username, email, password, confirm_password]):
            flash('All fields are required')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html')
        
        # Check if username or email already exists
        user_exists = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if user_exists:
            flash('Username or email already exists')
            return render_template('signup.html')
        
        # Create new user
        new_user = User(username=username, email=email, language_preference=language)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        # Automatically log in the user after account creation
        login_user(new_user)
        flash('Account created successfully! You are now logged in.')
        return redirect(url_for('main.dashboard'))
    
    return render_template('signup.html')

@main_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('main.index'))

# Dashboard routes
@main_bp.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get recent predictions for the user
        predictions = CropPrediction.query.filter_by(user_id=current_user.id).order_by(CropPrediction.created_at.desc()).limit(5).all()
        return render_template('dashboard.html', predictions=predictions)
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return render_template('dashboard.html', predictions=[])

@main_bp.route('/crop-dashboard')
@login_required  
def crop_dashboard():
    crops = ['All']
    if data_loaded and 'Crop' in df.columns:
        crops += sorted(df['Crop'].unique().tolist())
    return render_template('crop_dashboard.html', crops=crops, data_loaded=data_loaded)

@main_bp.route('/api/dashboard_data')
def api_dashboard_data():
    crop = request.args.get('crop', 'all')
    return jsonify(get_dashboard_data(crop))

# Crop prediction routes

@main_bp.route('/crop-prediction', methods=['GET', 'POST'])
@login_required
def crop_prediction():
    if request.method == 'POST':
        try:
            # Get only the required parameters (no crop_type, area, soil_type)
            nitrogen = float(request.form.get('nitrogen', 0))
            phosphorus = float(request.form.get('phosphorus', 0))
            potassium = float(request.form.get('potassium', 0))
            rainfall = float(request.form.get('rainfall', 0))
            temperature = float(request.form.get('temperature', 25))
            humidity = float(request.form.get('humidity', 70))
            ph = float(request.form.get('ph', 6.5))
            
            # Use ML model for prediction if available
            if ML_MODEL_AVAILABLE:
                try:
                    prediction_data = get_ml_prediction(
                        nitrogen=nitrogen,
                        phosphorus=phosphorus,
                        potassium=potassium,
                        rainfall=rainfall,
                        temperature=temperature,
                        humidity=humidity,
                        ph=ph
                    )
                except Exception as e:
                    print(f"ML model error: {str(e)}. Using fallback prediction.")
                    prediction_data = enhanced_crop_prediction(
                        nitrogen, phosphorus, potassium, rainfall, temperature, humidity, ph
                    )
            else:
                # Enhanced prediction model with multiple factors
                prediction_data = enhanced_crop_prediction(
                    nitrogen, phosphorus, potassium, rainfall, temperature, humidity, ph
                )
            
            # Save prediction to database
            prediction = CropPrediction(
                user_id=current_user.id,
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                rainfall=rainfall,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                suitability_score=prediction_data['suitability_score'],
                estimated_yield=prediction_data.get('estimated_yield', 0),
                recommended_crop=prediction_data.get('recommended_crop', 'Unknown'),
                crop_type=prediction_data.get('recommended_crop', 'Unknown'),
                season='Unknown',
                area=0.0,
                soil_type='Unknown',
                prediction_result=prediction_data.get('result', 'Unknown'),
                yield_estimation=prediction_data.get('yield_estimation', 'Unknown'),
                recommendations='; '.join(prediction_data.get('recommendations', []))
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            flash('Prediction generated successfully!', 'success')
            
            return render_template('crop_prediction.html',
                                  prediction=prediction_data,
                                  error=False)
        
        except ValueError as e:
            flash(f'Error in input values: {str(e)}', 'error')
            return render_template('crop_prediction.html',
                             error=True)
        except sqlalchemy.exc.IntegrityError as e:
            db.session.rollback()  # Rollback the failed transaction
            flash('Database integrity error. Please check your input values.', 'error')
            return render_template('crop_prediction.html',
                             error=True)
        except sqlalchemy.exc.OperationalError as e:
            db.session.rollback()  # Rollback the failed transaction
            flash('Database schema needs update. Prediction saved without database storage.', 'warning')
            
            # Still show the prediction result
            return render_template('crop_prediction.html',
                                  prediction=prediction_data,
                                  error=False)
        except Exception as e:
            db.session.rollback()  # Rollback any failed transaction
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('crop_prediction.html',
                             error=True)
    
    return render_template('crop_prediction_form.html')
@main_bp.route('/yield-estimation', methods=['GET', 'POST'])
@login_required
def yield_estimation():
    if request.method == 'POST':
        try:
            # Get only the required parameters (no crop_type, area, soil_type)
            nitrogen = float(request.form.get('nitrogen', 0))
            phosphorus = float(request.form.get('phosphorus', 0))
            potassium = float(request.form.get('potassium', 0))
            rainfall = float(request.form.get('rainfall', 0))
            temperature = float(request.form.get('temperature', 25))
            humidity = float(request.form.get('humidity', 70))
            ph = float(request.form.get('ph', 6.5))
            
            # Use ML model for prediction if available
            if ML_MODEL_AVAILABLE:
                try:
                    prediction_data = get_ml_prediction(
                        nitrogen=nitrogen,
                        phosphorus=phosphorus,
                        potassium=potassium,
                        rainfall=rainfall,
                        temperature=temperature,
                        humidity=humidity,
                        ph=ph
                    )
                except Exception as e:
                    print(f"ML model error: {str(e)}. Using fallback prediction.")
                    prediction_data = enhanced_crop_prediction(
                        nitrogen, phosphorus, potassium, rainfall, temperature, humidity, ph
                    )
            else:
                # Enhanced prediction model with multiple factors
                prediction_data = enhanced_crop_prediction(
                    nitrogen, phosphorus, potassium, rainfall, temperature, humidity, ph
                )
            
            # Save prediction to database
            prediction = CropPrediction(
                user_id=current_user.id,
                nitrogen=nitrogen,
                phosphorus=phosphorus,
                potassium=potassium,
                rainfall=rainfall,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                suitability_score=prediction_data['suitability_score'],
                estimated_yield=prediction_data.get('estimated_yield', 0),
                recommended_crop=prediction_data.get('recommended_crop', 'Unknown'),
                crop_type=prediction_data.get('recommended_crop', 'Unknown'),
                season='Unknown',
                area=0.0,
                soil_type='Unknown',
                prediction_result=prediction_data.get('result', 'Unknown'),
                yield_estimation=prediction_data.get('yield_estimation', 'Unknown'),
                recommendations='; '.join(prediction_data.get('recommendations', []))
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            flash('Yield estimation generated successfully!', 'success')
            
            return render_template('yield_estimation.html',
                                  yield_result=prediction_data,
                                  error=False)
        
        except ValueError as e:
            flash(f'Error in input values: {str(e)}', 'error')
            return render_template('yield_estimation.html',
                             yield_result={'result': f'Error in input values: {str(e)}'},
                             error=True)
        except sqlalchemy.exc.IntegrityError as e:
            db.session.rollback()  # Rollback the failed transaction
            flash('Database integrity error. Please check your input values.', 'error')
            return render_template('yield_estimation.html',
                             yield_result={'result': 'Database integrity error. Please check your input values.'},
                             error=True)
        except Exception as e:
            db.session.rollback()  # Rollback any failed transaction
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('yield_estimation.html',
                             yield_result={'result': f'An error occurred: {str(e)}'},
                             error=True)
    
    return render_template('yield_estimation.html')

# Prediction history routes
@main_bp.route('/predictions')
@login_required
def predictions():
    user_predictions = CropPrediction.query.filter_by(user_id=current_user.id).order_by(CropPrediction.created_at.desc()).all()
    
    # Prepare data for charts
    crop_counts = {}
    
    for prediction in user_predictions:
        crop = prediction.recommended_crop
        crop_counts[crop] = crop_counts.get(crop, 0) + 1
    
    crop_data = {
        'labels': list(crop_counts.keys()),
        'values': list(crop_counts.values())
    }
    
    return render_template('predictions.html',
                         predictions=user_predictions,
                         crop_data=crop_data)

@main_bp.route('/prediction/<int:prediction_id>')
@login_required
def prediction_detail(prediction_id):
    prediction = CropPrediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    return render_template('prediction_detail.html', prediction=prediction)

# Weather forecast route
@main_bp.route('/weather', methods=['GET', 'POST'])
@login_required
def weather():
    weather_data = None
    forecast_data = None
    error = None

    if request.method == "POST":
        city = request.form.get("location")
        try:
            # Current weather
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            weather_response = requests.get(weather_url).json()

            if weather_response.get("cod") != 200:
                error = weather_response.get("message", "City not found")
            else:
                weather_data = weather_response

                # Forecast (5 days, 3-hour interval)
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
                forecast_response = requests.get(forecast_url).json()

                # Process into daily averages
                labels, temps, humidity = [], [], []
                daily_data = {}

                for entry in forecast_response["list"]:
                    date = entry["dt_txt"].split(" ")[0]
                    if date not in daily_data:
                        daily_data[date] = {"temps": [], "humidity": []}
                    daily_data[date]["temps"].append(entry["main"]["temp"])
                    daily_data[date]["humidity"].append(entry["main"]["humidity"])

                for date, values in list(daily_data.items())[:5]:  # limit to 5 days
                    labels.append(date)
                    temps.append(sum(values["temps"]) / len(values["temps"]))
                    humidity.append(sum(values["humidity"]) / len(values["humidity"]))

                forecast_data = {
                    "labels": labels,
                    "temperatures": temps,
                    "humidity": humidity,
                }

        except Exception as e:
            error = str(e)

    return render_template(
        "weather.html",
        weather_data=weather_data,
        forecast_data=forecast_data,
        error=error,
    )

# Pest and disease management routes
@main_bp.route('/pest-guide')
@login_required
def pest_guide():
    pests = [
        {
            'name': 'Aphids',
            'crops': 'Cotton, vegetables, fruits',
            'symptoms': 'Curling of leaves, yellowing, stunted growth',
            'management': 'Neem oil spray, introduce ladybugs, insecticidal soap',
            'image': '/images/aphids.jpg'
        },
        {
            'name': 'Whiteflies',
            'crops': 'Tomatoes, Cotton, Vegetables',
            'symptoms': 'Yellowing leaves, sticky honeydew, sooty mold',
            'management': 'Yellow sticky traps, neem oil, biological control',
            'image': '/images/whiteflies.jpg'
        },
        {
            'name': 'Thrips',
            'crops': 'Onions, Peppers, Flowers',
            'symptoms': 'Silver streaks on leaves, black dots, stunted growth',
            'management': 'Blue sticky traps, predatory mites, insecticidal soap',
            'image': '/images/thrips.jpg'
        }
    ]
    return render_template('pest_guide.html', pests=pests)

@main_bp.route('/pest-question', methods=['POST'])
@login_required
def pest_question():
    pest_question = request.form.get('pest_question')
    crop_type = request.form.get('crop_type')
    urgent = request.form.get('urgent')
    
    # Save question to database or send to experts
    if urgent:
        flash('Your urgent question has been submitted! Our experts will prioritize your request.', 'success')
    else:
        flash('Your question has been submitted! Our experts will respond soon.', 'success')
    
    return redirect(url_for('main.pest_guide'))

# Pesticide analysis routes
@main_bp.route('/pesticide-analysis', methods=['GET', 'POST'])
@login_required
def pesticide_analysis():
    prediction = None
    error = None
    uploaded_filename = None
    analysis_stats = None

    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                error = 'No file part in request.'
            else:
                file = request.files['image']
                if file.filename == '':
                    error = 'No selected file.'
                elif file and allowed_file(file.filename):
                    # Secure the filename and add timestamp
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                    filename = timestamp + filename
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Save the uploaded file
                    file.save(filepath)
                    uploaded_filename = filename
                    
                    # Record analysis start time
                    analysis_start = time.time()
                    
                    # Analyze the pesticide image
                    print(f"Starting pesticide analysis for {filename}")
                    prediction = analyze_pesticide_with_ai(filepath, filename)
                    
                    # Calculate total analysis time
                    total_analysis_time = round(time.time() - analysis_start, 2)
                    
                    # Create analysis statistics
                    analysis_stats = {
                        'total_time': total_analysis_time,
                        'model_time': prediction.get('analysis_time', 'N/A'),
                        'method': prediction.get('analysis_method', 'Unknown'),
                        'api_status': prediction.get('api_status', 'Unknown'),
                        'file_size': round(os.path.getsize(filepath) / 1024, 2),  # KB
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    flash(f'Image analyzed successfully using {prediction.get("analysis_method", "AI")}!', 'success')
                    
                else:
                    error = 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WebP files only.'
                    
        except Exception as e:
            error = f'Error processing image: {str(e)}'
            print(f"Pesticide analysis error: {str(e)}")

    # Check API status for display
    api_status = None
    if MODEL_AVAILABLE:
        try:
            api_available, api_message = test_api_connection()
            api_status = {
                'available': api_available,
                'message': api_message,
                'model_type': 'OpenAI GPT-4 Vision' if api_available else 'Fallback Analysis'
            }
        except Exception as e:
            api_status = {
                'available': False,
                'message': f'Connection test failed: {str(e)}',
                'model_type': 'Fallback Analysis'
            }
    else:
        api_status = {
            'available': False,
            'message': 'OpenAI model not loaded',
            'model_type': 'Pattern Analysis'
        }

    return render_template('pesticide_analysis.html',
                         prediction=prediction,
                         error=error,
                         uploaded_filename=uploaded_filename,
                         analysis_stats=analysis_stats,
                         api_status=api_status)

@main_bp.route('/crop-problem-analysis', methods=['GET', 'POST'])
@login_required
def crop_problem_analysis():
    prediction = None
    error = None
    uploaded_filename = None

    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                error = 'No file part in request.'
            else:
                file = request.files['image']
                if file.filename == '':
                    error = 'No selected file.'
                elif file and allowed_file(file.filename):
                    # Secure the filename and add timestamp
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                    filename = timestamp + filename
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Save the uploaded file
                    file.save(filepath)
                    uploaded_filename = filename
                    
                    # Analyze the crop problem image
                    print(f"Starting crop problem analysis for {filename}")
                    if MODEL_AVAILABLE:
                        prediction = analyze_crop_problem_with_ai(filepath, filename)
                    else:
                        # Fallback analysis for crop problems
                        prediction = {
                            'problem': 'Unable to analyze crop problem',
                            'confidence': 50,
                            'severity': 'Unknown',
                            'recommendation': 'Please consult an agricultural expert for proper diagnosis.',
                            'possible_causes': ['Disease', 'Pest damage', 'Nutrient deficiency', 'Environmental stress'],
                            'filename': filename,
                            'analysis_method': 'Fallback Analysis',
                            'analysis_time': 0.1
                        }
                    
                    flash(f'Crop problem analyzed successfully!', 'success')
                    
                else:
                    error = 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WebP files only.'
                    
        except Exception as e:
            error = f'Error processing image: {str(e)}'
            print(f"Crop problem analysis error: {str(e)}")

    return render_template('crop_problem_analysis.html',
                         prediction=prediction,
                         error=error,
                         uploaded_filename=uploaded_filename)


@main_bp.route('/gov-schemes')
@login_required
def gov_schemes():
    schemes = [
        {
            'name': 'Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)',
            'description': 'Income support of Rs. 6000 per year to farmer families',
            'eligibility': 'All small and marginal farmers with cultivable land',
            'link': 'https://pmkisan.gov.in/'
        },
        {
            'name': 'Pradhan Mantri Fasal Bima Yojana (PMFBY)',
            'description': 'Crop insurance scheme with minimal premium',
            'eligibility': 'All farmers growing notified crops',
            'link': 'https://pmfby.gov.in/'
        },
        {
            'name': 'Kisan Credit Card (KCC)',
            'description': 'Credit facility for farmers with minimal interest rate',
            'eligibility': 'All farmers, sharecroppers, and tenant farmers',
            'link': 'https://www.pnbindia.in/kisan-credit-card.html'
        },
        {
            'name': 'Soil Health Card Scheme',
            'description': 'Soil testing and recommendations for nutrient application',
            'eligibility': 'All farmers',
            'link': 'https://www.soilhealth.dac.gov.in/'
        },
        {
            'name': 'National Agriculture Market (e-NAM)',
            'description': 'Online trading platform for agricultural commodities',
            'eligibility': 'All farmers with registered accounts',
            'link': 'https://www.enam.gov.in/'
        }
    ]
    
    return render_template('gov_schemes.html', schemes=schemes)

@main_bp.route('/check-eligibility', methods=['POST'])
@login_required
def check_eligibility():
    """Check eligibility for various agricultural schemes based on user input"""
    land_holding = float(request.form.get('land_holding', 0))
    farmer_category = request.form.get('farmer_category')
    state = request.form.get('state')
    
    # Convert acres to hectares (1 acre = 0.4047 hectares)
    land_hectares = land_holding * 0.4047
    
    eligible_schemes = []
    
    # PM-KISAN - All landholding farmers
    if land_holding > 0:
        eligible_schemes.append({
            'name': 'PM-KISAN',
            'description': 'Income support of ₹6,000 per year for all landholding farmers'
        })
    
    # Kisan Credit Card - All farmers
    eligible_schemes.append({
        'name': 'Kisan Credit Card (KCC)',
        'description': 'Credit facility for agricultural needs at subsidized interest rates'
    })
    
    # PMFBY - All farmers growing notified crops
    eligible_schemes.append({
        'name': 'Pradhan Mantri Fasal Bima Yojana (PMFBY)',
        'description': 'Crop insurance with low premium rates'
    })
    
    # Schemes based on farmer category
    if farmer_category in ['marginal', 'small']:
        eligible_schemes.append({
            'name': 'PMKSY - Micro Irrigation',
            'description': 'Higher subsidy rates for micro-irrigation systems for small/marginal farmers'
        })
        
        eligible_schemes.append({
            'name': 'Sub-Mission on Agricultural Mechanization',
            'description': 'Higher subsidy for agricultural machinery and equipment'
        })
    
    # Soil Health Card - All farmers
    eligible_schemes.append({
        'name': 'Soil Health Card Scheme',
        'description': 'Free soil testing and nutrient recommendations'
    })
    
    flash(f'Found {len(eligible_schemes)} eligible schemes for your profile!', 'success')
    return render_template('gov_schemes.html', 
                         schemes=[], 
                         eligible_schemes=eligible_schemes,
                         land_holding=land_holding,
                         farmer_category=farmer_category,
                         state=state)

# Market trends route
@main_bp.route('/market-trends')
@login_required
def market_trends():
    return render_template('market_trends.html')

# API routes
@main_bp.route('/api/analyze-pesticide', methods=['POST'])
@login_required
def api_analyze_pesticide():
    """Enhanced API endpoint for pesticide analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Analyze image
        analysis_start = time.time()
        result = analyze_pesticide_with_ai(filepath, filename)
        total_time = round(time.time() - analysis_start, 2)
        
        # Add API-specific metadata
        api_response = {
            'success': True,
            'result': result,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_analysis_time': total_time,
                'file_size_kb': round(os.path.getsize(filepath) / 1024, 2),
                'analysis_method': result.get('analysis_method', 'Unknown'),
                'api_version': '2.0',
                'model_available': MODEL_AVAILABLE
            }
        }
        
        return jsonify(api_response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@main_bp.route('/api/status')
def api_status():
    """Check API and model status"""
    status_info = {
        'service': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'openai_model_available': MODEL_AVAILABLE,
        'ml_model_available': ML_MODEL_AVAILABLE,
        'data_loaded': data_loaded
    }
    
    if MODEL_AVAILABLE:
        try:
            api_available, api_message = test_api_connection()
            status_info.update({
                'openai_api_available': api_available,
                'openai_api_message': api_message,
                'analysis_methods': ['OpenAI GPT-4 Vision', 'Pattern Analysis', 'Emergency Fallback']
            })
        except Exception as e:
            status_info.update({
                'openai_api_available': False,
                'openai_api_message': f'Connection test failed: {str(e)}',
                'analysis_methods': ['Pattern Analysis', 'Emergency Fallback']
            })
    else:
        status_info.update({
            'openai_api_available': False,
            'openai_api_message': 'Model not loaded',
            'analysis_methods': ['Emergency Fallback']
        })
    
    return jsonify(status_info)

# Health check route
@main_bp.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_model_available": MODEL_AVAILABLE,
        "ml_model_available": ML_MODEL_AVAILABLE,
        "data_loaded": data_loaded,
        "version": "2.0"
    })