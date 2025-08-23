

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropPredictionModel:
    def __init__(self, model_dir='app/ml_models'):
        self.model_dir = model_dir
        self.suitability_model = None
        self.yield_model = None
        self.label_encoders = {}
        self.scaler = None
        self.is_trained = False
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing models
        self.load_models()
    
    def prepare_training_data(self):
        """
        Prepare synthetic training data for the ML model
        In a real implementation, this would load from a database or CSV files
        """
        
        # Synthetic training data based on agricultural research
        training_data = []
        
        # Define crop characteristics
        crop_data = {
            'rice': {
                'temp_range': (20, 35), 'rainfall_range': (1000, 2000),
                'nitrogen': (80, 120), 'phosphorus': (40, 60), 'potassium': (40, 60),
                'base_yield': 4.0, 'seasons': ['kharif'], 'soils': ['clay', 'loam']
            },
            'wheat': {
                'temp_range': (15, 25), 'rainfall_range': (400, 1100),
                'nitrogen': (100, 120), 'phosphorus': (50, 60), 'potassium': (40, 50),
                'base_yield': 3.5, 'seasons': ['rabi'], 'soils': ['loam', 'clay', 'silt']
            },
            'maize': {
                'temp_range': (21, 30), 'rainfall_range': (500, 1000),
                'nitrogen': (120, 150), 'phosphorus': (60, 80), 'potassium': (60, 80),
                'base_yield': 5.2, 'seasons': ['kharif', 'rabi'], 'soils': ['loam', 'sandy']
            },
            'cotton': {
                'temp_range': (21, 30), 'rainfall_range': (500, 1000),
                'nitrogen': (100, 150), 'phosphorus': (50, 80), 'potassium': (50, 80),
                'base_yield': 1.8, 'seasons': ['kharif'], 'soils': ['black', 'loam']
            },
            'sugarcane': {
                'temp_range': (26, 32), 'rainfall_range': (1000, 1500),
                'nitrogen': (200, 300), 'phosphorus': (80, 120), 'potassium': (100, 150),
                'base_yield': 70.0, 'seasons': ['perennial'], 'soils': ['loam', 'clay']
            }
        }
        
        # Generate synthetic training samples
        np.random.seed(42)  # For reproducibility
        
        for crop_name, crop_info in crop_data.items():
            for _ in range(200):  # Generate 200 samples per crop
                # Random variations around optimal conditions
                temp_var = np.random.normal(0, 3)
                rain_var = np.random.normal(0, 100)
                
                temperature = np.random.uniform(*crop_info['temp_range']) + temp_var
                rainfall = np.random.uniform(*crop_info['rainfall_range']) + rain_var
                
                # Nutrient variations
                nitrogen = np.random.uniform(*crop_info['nitrogen']) + np.random.normal(0, 10)
                phosphorus = np.random.uniform(*crop_info['phosphorus']) + np.random.normal(0, 5)
                potassium = np.random.uniform(*crop_info['potassium']) + np.random.normal(0, 5)
                
                # Random area
                area = np.random.uniform(0.5, 10.0)
                
                # Random season and soil from preferred options
                season = np.random.choice(crop_info['seasons'])
                soil_type = np.random.choice(crop_info['soils'])
                
                # Calculate suitability based on conditions
                temp_score = self.calculate_condition_score(
                    temperature, crop_info['temp_range']
                )
                rain_score = self.calculate_condition_score(
                    rainfall, crop_info['rainfall_range']
                )
                
                # Overall suitability (0-100)
                suitability = (temp_score + rain_score) / 2
                suitability += np.random.normal(0, 5)  # Add some noise
                suitability = np.clip(suitability, 0, 100)
                
                # Calculate yield based on suitability and conditions
                yield_modifier = suitability / 100
                base_yield = crop_info['base_yield']
                estimated_yield = base_yield * yield_modifier * area
                estimated_yield += np.random.normal(0, estimated_yield * 0.1)  # 10% noise
                estimated_yield = max(0, estimated_yield)
                
                training_data.append({
                    'crop_type': crop_name,
                    'season': season,
                    'area': area,
                    'soil_type': soil_type,
                    'nitrogen': max(0, nitrogen),
                    'phosphorus': max(0, phosphorus),
                    'potassium': max(0, potassium),
                    'rainfall': max(0, rainfall),
                    'temperature': temperature,
                    'suitability_score': suitability,
                    'estimated_yield': estimated_yield
                })
        
        return pd.DataFrame(training_data)
    
    def calculate_condition_score(self, value, optimal_range):
        """Calculate score based on how close value is to optimal range"""
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 100
        elif value < min_val:
            deviation = (min_val - value) / min_val
            return max(0, 100 - (deviation * 100))
        else:
            deviation = (value - max_val) / max_val
            return max(0, 100 - (deviation * 100))
    
    def train_models(self):
        """Train the ML models"""
        logger.info("Preparing training data...")
        df = self.prepare_training_data()
        
        # Prepare features
        categorical_features = ['crop_type', 'season', 'soil_type']
        numerical_features = ['area', 'nitrogen', 'phosphorus', 'potassium', 'rainfall', 'temperature']
        
        # Encode categorical variables
        df_encoded = df.copy()
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            df_encoded[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        # Prepare feature matrix
        feature_columns = categorical_features + numerical_features
        X = df_encoded[feature_columns]
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Train suitability model (classification/regression)
        logger.info("Training suitability model...")
        y_suitability = df['suitability_score']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_suitability, test_size=0.2, random_state=42
        )
        
        self.suitability_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.suitability_model.fit(X_train, y_train)
        
        # Evaluate suitability model
        y_pred = self.suitability_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Suitability model MSE: {mse:.2f}")
        
        # Train yield model
        logger.info("Training yield model...")
        y_yield = df['estimated_yield']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_yield, test_size=0.2, random_state=42
        )
        
        self.yield_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.yield_model.fit(X_train, y_train)
        
        # Evaluate yield model
        y_pred = self.yield_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Yield model MSE: {mse:.2f}")
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        logger.info("Models trained and saved successfully!")
    
    def save_models(self):
        """Save trained models to disk"""
        if not self.is_trained:
            return
        
        try:
            # Save models
            joblib.dump(self.suitability_model, os.path.join(self.model_dir, 'suitability_model.pkl'))
            joblib.dump(self.yield_model, os.path.join(self.model_dir, 'yield_model.pkl'))
            joblib.dump(self.label_encoders, os.path.join(self.model_dir, 'label_encoders.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_version': '1.0',
                'features': ['crop_type', 'season', 'soil_type', 'area', 'nitrogen', 
                           'phosphorus', 'potassium', 'rainfall', 'temperature']
            }
            
            with open(os.path.join(self.model_dir, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info("Models saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            suitability_path = os.path.join(self.model_dir, 'suitability_model.pkl')
            yield_path = os.path.join(self.model_dir, 'yield_model.pkl')
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            
            if all(os.path.exists(path) for path in [suitability_path, yield_path, encoders_path, scaler_path]):
                self.suitability_model = joblib.load(suitability_path)
                self.yield_model = joblib.load(yield_path)
                self.label_encoders = joblib.load(encoders_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                logger.info("Models loaded successfully!")
            else:
                logger.info("No pre-trained models found. Will need to train.")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_trained = False
    
    def predict(self, crop_type, season, area, soil_type, nitrogen, phosphorus, potassium, rainfall, temperature):
        """
        Make prediction using trained ML models
        """
        if not self.is_trained:
            logger.warning("Models not trained. Training now...")
            self.train_models()
        
        try:
            # Prepare input data
            input_data = {
                'crop_type': crop_type,
                'season': season,
                'soil_type': soil_type,
                'area': area,
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'rainfall': rainfall,
                'temperature': temperature
            }
            
            # Create DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Encode categorical variables
            categorical_features = ['crop_type', 'season', 'soil_type']
            for feature in categorical_features:
                if feature in self.label_encoders:
                    try:
                        df_input[feature] = self.label_encoders[feature].transform(df_input[feature])
                    except ValueError:
                        # Handle unknown categories
                        logger.warning(f"Unknown category for {feature}: {input_data[feature]}")
                        df_input[feature] = 0  # Default to first category
            
            # Scale numerical features
            numerical_features = ['area', 'nitrogen', 'phosphorus', 'potassium', 'rainfall', 'temperature']
            df_scaled = df_input.copy()
            df_scaled[numerical_features] = self.scaler.transform(df_input[numerical_features])
            
            # Make predictions
            suitability_score = self.suitability_model.predict(df_scaled)[0]
            estimated_yield = self.yield_model.predict(df_scaled)[0]
            
            # Ensure reasonable bounds
            suitability_score = np.clip(suitability_score, 0, 100)
            estimated_yield = max(0, estimated_yield)
            
            # Generate recommendations based on predictions
            recommendations = self.generate_recommendations(
                crop_type, suitability_score, input_data
            )
            
            # Create result message
            suitability_text = (
                "Excellent" if suitability_score >= 90 else 
                "Good" if suitability_score >= 75 else 
                "Fair" if suitability_score >= 60 else "Poor"
            )
            
            result = (
                f"Based on your field conditions, {crop_type} cultivation shows "
                f"{suitability_text.lower()} suitability ({suitability_score:.1f}%) "
                f"for {season} season in {area} hectares."
            )
            
            yield_estimation = (
                f"Estimated yield: {estimated_yield:.2f} tons "
                f"(Average: {estimated_yield/area:.2f} tons/hectare)"
            )
            
            return {
                'result': result,
                'yield_estimation': yield_estimation,
                'suitability_score': round(suitability_score, 1),
                'recommendations': recommendations,
                'prediction_method': 'Machine Learning Model'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Fallback to simple prediction
            return self.fallback_prediction(crop_type, season, area)
    
    def generate_recommendations(self, crop_type, suitability_score, input_data):
        """Generate recommendations based on prediction results"""
        recommendations = []
        
        if suitability_score < 70:
            recommendations.append(
                f"The current conditions show suboptimal suitability for {crop_type}. "
                "Consider soil amendments or alternative crops."
            )
        
        if input_data['nitrogen'] < 50:
            recommendations.append("Consider increasing nitrogen application for better growth.")
        
        if input_data['phosphorus'] < 30:
            recommendations.append("Phosphorus levels appear low. Consider phosphate fertilizers.")
        
        if input_data['potassium'] < 30:
            recommendations.append("Potassium supplementation may improve crop resilience.")
        
        if input_data['rainfall'] < 500:
            recommendations.append("Low rainfall conditions. Ensure adequate irrigation.")
        
        if not recommendations:
            recommendations.append("Current conditions are suitable for the selected crop.")
        
        return recommendations
    
    def fallback_prediction(self, crop_type, season, area):
        """Fallback prediction when ML model fails"""
        return {
            'result': f"Basic prediction for {crop_type} in {season} season for {area} hectares.",
            'yield_estimation': f"Estimated yield: {3.0 * area:.2f} tons",
            'suitability_score': 75.0,
            'recommendations': ["Machine learning model unavailable. Using basic estimation."],
            'prediction_method': 'Fallback Method'
        }


# Global model instance
ml_model = CropPredictionModel()

def get_ml_prediction(crop_type, season, area, soil_type, nitrogen, phosphorus, potassium, rainfall, temperature):
    """
    Main function to get ML-based crop prediction
    """
    return ml_model.predict(
        crop_type, season, area, soil_type, 
        nitrogen, phosphorus, potassium, rainfall, temperature
    )

def train_models_if_needed():
    """Train models if they haven't been trained yet"""
    if not ml_model.is_trained:
        logger.info("Training ML models...")
        ml_model.train_models()
    return ml_model.is_trained