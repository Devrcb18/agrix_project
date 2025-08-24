import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropPredictionModel:
    def __init__(self, model_dir='ml_models', data_path='files/Crop_recommendation.csv'):
        self.model_dir = model_dir
        self.data_path = data_path
        self.suitability_model = None
        self.yield_model = None
        self.scaler = None
        self.is_trained = False
        self.crop_optimal_ranges = {}
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing models
        self.load_models()
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the actual crop recommendation dataset
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Display basic info about the dataset
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Check the target variable (assuming it's 'label' for crop type)
            if 'label' in df.columns:
                logger.info(f"Target variable 'label' has {df['label'].nunique()} unique crops")
                logger.info(f"Unique crops: {sorted(df['label'].unique())}")
            
            # Rename columns to match our expected feature names
            column_mapping = {
                'N': 'nitrogen',
                'P': 'phosphorus', 
                'K': 'potassium',
                'temperature': 'temperature',
                'humidity': 'humidity',
                'ph': 'ph',
                'rainfall': 'rainfall',
                'label': 'crop_type'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Calculate optimal ranges for each crop
            self.calculate_crop_optimal_ranges(df)
            logger.info(f"Calculated optimal ranges for {len(self.crop_optimal_ranges)} crops")
            
            # Calculate suitability score for each sample
            df['suitability_score'] = df.apply(
                lambda row: self.calculate_suitability_score(
                    row['crop_type'],
                    row['nitrogen'],
                    row['phosphorus'],
                    row['potassium'],
                    row['temperature'],
                    row['humidity'],
                    row['ph'],
                    row['rainfall']
                ), 
                axis=1
            )
            
            # Calculate estimated yield based on suitability
            df['estimated_yield'] = df.apply(
                lambda row: self.calculate_estimated_yield(
                    row['crop_type'],
                    row['suitability_score']
                ), 
                axis=1
            )
            
            # Select only the numerical features we need
            final_columns = [
                'nitrogen', 'phosphorus', 'potassium', 'temperature', 
                'humidity', 'ph', 'rainfall', 'suitability_score', 'estimated_yield'
            ]
            
            df_processed = df[final_columns]
            
            logger.info(f"Processed dataset shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.error(f"Looking for file at: {os.path.abspath(self.data_path)}")
            # Fall back to synthetic data if real dataset is unavailable
            logger.info("Falling back to synthetic data...")
            return self.prepare_training_data()
    
    def calculate_crop_optimal_ranges(self, df):
        """
        Calculate optimal ranges for each crop based on dataset statistics
        """
        for crop in df['crop_type'].unique():
            crop_data = df[df['crop_type'] == crop]
            
            self.crop_optimal_ranges[crop] = {
                'nitrogen': (crop_data['nitrogen'].quantile(0.25), crop_data['nitrogen'].quantile(0.75)),
                'phosphorus': (crop_data['phosphorus'].quantile(0.25), crop_data['phosphorus'].quantile(0.75)),
                'potassium': (crop_data['potassium'].quantile(0.25), crop_data['potassium'].quantile(0.75)),
                'temperature': (crop_data['temperature'].quantile(0.25), crop_data['temperature'].quantile(0.75)),
                'humidity': (crop_data['humidity'].quantile(0.25), crop_data['humidity'].quantile(0.75)),
                'ph': (crop_data['ph'].quantile(0.25), crop_data['ph'].quantile(0.75)),
                'rainfall': (crop_data['rainfall'].quantile(0.25), crop_data['rainfall'].quantile(0.75))
            }
    
    def calculate_suitability_score(self, crop_type, nitrogen, phosphorus, potassium, 
                                  temperature, humidity, ph, rainfall):
        """
        Calculate suitability score based on how close values are to crop's optimal ranges
        """
        if crop_type not in self.crop_optimal_ranges:
            return 70  # Default score for unknown crops
        
        optimal_ranges = self.crop_optimal_ranges[crop_type]
        total_score = 0
        factors_considered = 0
        
        # Check each factor
        factors = {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        for factor, value in factors.items():
            if factor in optimal_ranges:
                min_val, max_val = optimal_ranges[factor]
                
                if min_val <= value <= max_val:
                    total_score += 100
                else:
                    # Calculate deviation penalty
                    if value < min_val:
                        deviation = (min_val - value) / min_val
                    else:
                        deviation = (value - max_val) / max_val
                    total_score += max(0, 100 - (deviation * 100))
                
                factors_considered += 1
        
        return total_score / factors_considered if factors_considered > 0 else 70
    
    def calculate_estimated_yield(self, crop_type, suitability_score):
        """
        Calculate estimated yield based on crop type and suitability score
        """
        # Base yields for different crops (hypothetical values in tons/hectare)
        base_yields = {
            'rice': 4.0, 'maize': 5.2, 'chickpea': 2.5, 'kidneybeans': 2.0,
            'pigeonpeas': 2.2, 'mothbeans': 1.8, 'mungbean': 2.1, 'blackgram': 2.0,
            'lentil': 1.8, 'pomegranate': 8.0, 'banana': 12.0, 'mango': 6.0,
            'grapes': 5.0, 'watermelon': 10.0, 'muskmelon': 8.0, 'apple': 6.0,
            'orange': 7.0, 'papaya': 15.0, 'coconut': 20.0, 'cotton': 1.8,
            'jute': 2.5, 'coffee': 1.5
        }
        
        base_yield = base_yields.get(crop_type, 3.0)
        return base_yield * (suitability_score / 100) * np.random.uniform(0.9, 1.1)
    
    def prepare_training_data(self):
        """
        Fallback synthetic training data if real dataset is unavailable
        """
        # Simplified synthetic data without crop, area, soil type
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'nitrogen': np.random.uniform(0, 200, n_samples),
            'phosphorus': np.random.uniform(0, 150, n_samples),
            'potassium': np.random.uniform(0, 200, n_samples),
            'temperature': np.random.uniform(10, 40, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'ph': np.random.uniform(4.0, 9.0, n_samples),
            'rainfall': np.random.uniform(200, 2000, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate suitability score (simplified)
        df['suitability_score'] = df.apply(
            lambda row: self.calculate_general_suitability(
                row['nitrogen'], row['phosphorus'], row['potassium'],
                row['temperature'], row['humidity'], row['ph'], row['rainfall']
            ), 
            axis=1
        )
        
        # Calculate estimated yield
        df['estimated_yield'] = df['suitability_score'] / 100 * np.random.uniform(2.0, 8.0, n_samples)
        
        return df
    
    def calculate_general_suitability(self, nitrogen, phosphorus, potassium, 
                                    temperature, humidity, ph, rainfall):
        """
        General suitability calculation for synthetic data
        """
        # Define general optimal ranges
        optimal_ranges = {
            'nitrogen': (50, 120),
            'phosphorus': (30, 80),
            'potassium': (40, 100),
            'temperature': (20, 30),
            'humidity': (60, 80),
            'ph': (6.0, 7.0),
            'rainfall': (500, 1200)
        }
        
        total_score = 0
        factors = {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        
        for factor, value in factors.items():
            min_val, max_val = optimal_ranges[factor]
            
            if min_val <= value <= max_val:
                total_score += 100
            else:
                if value < min_val:
                    deviation = (min_val - value) / min_val
                else:
                    deviation = (value - max_val) / max_val
                total_score += max(0, 100 - (deviation * 100))
        
        return total_score / len(factors)
    
    def train_models(self):
        """Train the ML models using the actual dataset"""
        logger.info("Loading and preparing training data...")
        df = self.load_and_preprocess_data()
        
        # Prepare features (only numerical features now)
        numerical_features = ['nitrogen', 'phosphorus', 'potassium', 
                             'temperature', 'humidity', 'ph', 'rainfall']
        
        # Prepare feature matrix
        X = df[numerical_features]
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train suitability model
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
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Suitability model MSE: {mse:.2f}, R²: {r2:.2f}")
        
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
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Yield model MSE: {mse:.2f}, R²: {r2:.2f}")
        
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
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.crop_optimal_ranges, os.path.join(self.model_dir, 'crop_ranges.pkl'))
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_version': '1.0',
                'features': ['nitrogen', 'phosphorus', 'potassium', 
                           'temperature', 'humidity', 'ph', 'rainfall']
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
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            ranges_path = os.path.join(self.model_dir, 'crop_ranges.pkl')
            
            if all(os.path.exists(path) for path in [suitability_path, yield_path, scaler_path, ranges_path]):
                self.suitability_model = joblib.load(suitability_path)
                self.yield_model = joblib.load(yield_path)
                self.scaler = joblib.load(scaler_path)
                self.crop_optimal_ranges = joblib.load(ranges_path)
                self.is_trained = True
                logger.info("Models loaded successfully!")
            else:
                logger.info("No pre-trained models found. Will need to train.")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.is_trained = False
    
    def predict(self, nitrogen, phosphorus, potassium, rainfall, temperature, humidity=70, ph=6.5):
        """
        Make prediction using trained ML models
        Only environmental and soil nutrient parameters as input
        """
        if not self.is_trained:
            logger.warning("Models not trained. Training now...")
            self.train_models()
        
        try:
            # Prepare input data
            input_data = {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # Create DataFrame with feature names to preserve column order
            feature_names = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
            input_features = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                                          columns=feature_names)
            
            # Scale input features
            input_scaled = self.scaler.transform(input_features)
            
            # Make predictions
            suitability_score = self.suitability_model.predict(input_scaled)[0]
            estimated_yield = self.yield_model.predict(input_scaled)[0]
            
            # Ensure reasonable bounds
            suitability_score = np.clip(suitability_score, 0, 100)
            estimated_yield = max(0, estimated_yield)
            
            # Find the most suitable crop based on conditions
            recommended_crop = self.find_recommended_crop(input_data)
            
            # Generate recommendations based on predictions
            recommendations = self.generate_recommendations(suitability_score, input_data)
            
            # Create result message
            suitability_text = (
                "Excellent" if suitability_score >= 90 else 
                "Good" if suitability_score >= 75 else 
                "Fair" if suitability_score >= 60 else "Poor"
            )
            
            result = (
                f"Based on your field conditions, the suitability for crop cultivation is "
                f"{suitability_text.lower()} ({suitability_score:.1f}%). "
                f"Recommended crop: {recommended_crop}"
            )
            
            yield_estimation = (
                f"Estimated yield: {estimated_yield:.2f} tons/hectare"
            )
            
            return {
                'result': result,
                'yield_estimation': yield_estimation,
                'suitability_score': round(suitability_score, 1),
                'recommended_crop': recommended_crop,
                'recommendations': recommendations,
                'prediction_method': 'Machine Learning Model'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Fallback to simple prediction
            return self.fallback_prediction()
    
    def find_recommended_crop(self, input_data):
        """
        Find the most suitable crop based on input conditions
        """
        best_crop = "rice"  # Default to rice instead of "General Crops"
        best_score = 0
        
        # If we don't have crop optimal ranges, return a sensible default based on conditions
        if not self.crop_optimal_ranges:
            # Simple heuristic based on temperature and rainfall
            temp = input_data.get('temperature', 25)
            rainfall = input_data.get('rainfall', 1000)
            
            if temp > 30 and rainfall > 1500:
                return "rice"
            elif temp < 20 and rainfall < 600:
                return "wheat"
            elif 20 <= temp <= 30:
                return "maize"
            else:
                return "rice"
        
        for crop, ranges in self.crop_optimal_ranges.items():
            score = 0
            factors_considered = 0
            
            for factor, value in input_data.items():
                if factor in ranges:
                    min_val, max_val = ranges[factor]
                    if min_val <= value <= max_val:
                        score += 100
                    else:
                        if value < min_val:
                            deviation = (min_val - value) / min_val if min_val > 0 else 0
                        else:
                            deviation = (value - max_val) / max_val if max_val > 0 else 0
                        score += max(0, 100 - (deviation * 100))
                    factors_considered += 1
            
            if factors_considered > 0:
                average_score = score / factors_considered
                if average_score > best_score:
                    best_score = average_score
                    best_crop = crop
        
        # If no good match found (all crops scored poorly), use temperature-based recommendation
        if best_score < 30:
            temp = input_data.get('temperature', 25)
            rainfall = input_data.get('rainfall', 1000)
            
            if temp > 30:
                return "rice" if rainfall > 1000 else "cotton"
            elif temp < 15:
                return "apple" if rainfall > 800 else "wheat"
            elif 15 <= temp <= 25:
                return "maize" if rainfall > 600 else "chickpea"
            else:
                return "rice"
        
        return best_crop
    
    def generate_recommendations(self, suitability_score, input_data):
        """Generate recommendations based on prediction results"""
        recommendations = []
        
        if suitability_score < 70:
            recommendations.append(
                "The current conditions show suboptimal suitability for most crops. "
                "Consider soil amendments or consulting an agricultural expert."
            )
        
        if input_data['nitrogen'] < 50:
            recommendations.append("Consider increasing nitrogen application for better growth.")
        elif input_data['nitrogen'] > 120:
            recommendations.append("Nitrogen levels are high. Consider reducing application to avoid environmental impact.")
        
        if input_data['phosphorus'] < 30:
            recommendations.append("Phosphorus levels appear low. Consider phosphate fertilizers.")
        
        if input_data['potassium'] < 40:
            recommendations.append("Potassium supplementation may improve crop resilience.")
        
        if input_data['rainfall'] < 500:
            recommendations.append("Low rainfall conditions. Ensure adequate irrigation.")
        elif input_data['rainfall'] > 1500:
            recommendations.append("High rainfall conditions. Ensure proper drainage to prevent waterlogging.")
        
        if input_data['ph'] < 6.0:
            recommendations.append("Soil pH is acidic. Consider lime application.")
        elif input_data['ph'] > 7.5:
            recommendations.append("Soil pH is alkaline. Consider sulfur or organic matter application.")
        
        if not recommendations:
            recommendations.append("Current conditions are suitable for most crops. Maintain good agricultural practices.")
        
        return recommendations
    
    def fallback_prediction(self):
        """Fallback prediction when ML model fails"""
        return {
            'result': "Basic prediction based on general agricultural conditions.",
            'yield_estimation': "Estimated yield: 3.0-5.0 tons/hectare",
            'suitability_score': 75.0,
            'recommended_crop': "rice",  # Default to rice instead of "General Crops"
            'recommendations': ["Machine learning model unavailable. Using basic estimation."],
            'prediction_method': 'Fallback Method'
        }


# Global model instance
ml_model = CropPredictionModel()

def get_ml_prediction(nitrogen, phosphorus, potassium, rainfall, temperature, humidity=70, ph=6.5):
    """
    Main function to get ML-based crop prediction
    Now only takes environmental and soil nutrient parameters
    """
    return ml_model.predict(nitrogen, phosphorus, potassium, rainfall, temperature, humidity, ph)

def train_models_if_needed():
    """Train models if they haven't been trained yet"""
    if not ml_model.is_trained:
        logger.info("Training ML models...")
        ml_model.train_models()
    return ml_model.is_trained