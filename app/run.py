import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from __init__ import create_app, db
from models import User, CropPrediction

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'CropPrediction': CropPrediction}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)