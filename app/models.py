from __init__ import db
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    language_preference = db.Column(db.String(20), default='english')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('CropPrediction', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class CropPrediction(db.Model):
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Input parameters
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    ph = db.Column(db.Float)
    
    # Prediction results
    suitability_score = db.Column(db.Float)
    estimated_yield = db.Column(db.Float)
    recommended_crop = db.Column(db.String(100))
    recommendations = db.Column(db.Text)
    
    # Legacy fields (keep for backward compatibility)
    crop_type = db.Column(db.String(100), nullable=False)
    season = db.Column(db.String(50))
    area = db.Column(db.Float)
    soil_type = db.Column(db.String(50))
    prediction_result = db.Column(db.Text)
    yield_estimation = db.Column(db.Text)
    
    # user relationship is handled by the backref in User model
    
    def __repr__(self):
        return f'<CropPrediction {self.recommended_crop} for {self.user_id}>'