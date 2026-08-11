from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/api/hello")
def hello():
    return jsonify({
        "message": "Face Recognition API is working!"
    })
