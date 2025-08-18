# src/main.py
import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.image import image_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

# register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(image_bp, url_prefix='/api')
print("📢 image_bp registered")

# --- debugging helpers (اضف بعد تسجيل البلوبيرينتات مباشرة) ---
print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(rule, "->", ",".join(sorted(rule.methods)))

@app.before_request
def log_request_info():
    print(">>> REQUEST:", request.method, request.path)

@app.errorhandler(405)
def handle_405(e):
    print("!!! 405 for:", request.method, request.path)
    return "Method Not Allowed", 405

@app.route('/api/test', methods=['GET', 'POST'])
def api_test():
    if request.method == 'POST':
        data = request.get_json(silent=True)
        print("api/test POST received, payload:", data)
        return {"ok": True, "method": "POST", "received": data}, 200
    return {"ok": True, "method": "GET"}, 200
# -------------------------------------------------------------------

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
