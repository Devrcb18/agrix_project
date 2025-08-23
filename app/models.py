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
    recommended_crop = db.Column(db.String(64), nullable=True)
    prediction_result = db.Column(db.Text, nullable=False)
    yield_estimation = db.Column(db.Text, nullable=True)
    suitability_score = db.Column(db.Float, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    nitrogen = db.Column(db.Float, nullable=True)
    phosphorus = db.Column(db.Float, nullable=True)
    potassium = db.Column(db.Float, nullable=True)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    rainfall = db.Column(db.Float, nullable=True)
    ph = db.Column(db.Float, nullable=True)
    estimated_yield = db.Column(db.Float, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<CropPrediction {self.recommended_crop} for {self.user_id}>'