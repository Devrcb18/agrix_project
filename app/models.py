from __init__ import db
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
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
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    crop_type = db.Column(db.String(64), nullable=False)
    season = db.Column(db.String(20), nullable=False)
    area = db.Column(db.Float, nullable=False)  
    prediction_result = db.Column(db.Text, nullable=False)
    yield_estimation = db.Column(db.Text, nullable=True)  
    soil_type = db.Column(db.String(100), nullable=True)
    nitrogen = db.Column(db.Float, nullable=True)
    phosphorus = db.Column(db.Float, nullable=True)
    potassium = db.Column(db.Float, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    rainfall = db.Column(db.Float, nullable=True)
    ph = db.Column(db.Float, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    expected_yield = db.Column(db.Float, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<CropPrediction {self.crop_type} for {self.user_id}>'