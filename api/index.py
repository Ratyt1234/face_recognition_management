import base64
import os

import face_recognition
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

known_encodings = []
known_names = []


def load_known_faces():
    global known_encodings, known_names

    known_encodings = []
    known_names = []

    folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "known_faces"
    )

    if not os.path.exists(folder):
        return

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(folder, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])

                name = os.path.splitext(filename)[0]
                known_names.append(name)

        except Exception as e:
            print("Error loading:", filename, e)


load_known_faces()


@app.route("/api/hello")
def hello():
    return jsonify({
        "message": "Face Recognition API is working!",
        "known_faces": known_names
    })


@app.route("/api/recognize", methods=["POST"])
def recognize():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "recognized": False,
                "error": "No image received"
            }), 400

        image_data = data["image"]

        # Remove data:image/jpeg;base64,...
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = face_recognition.load_image_file(
            __import__("io").BytesIO(image_bytes)
        )

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return jsonify({
                "recognized": False,
                "message": "No face detected"
            })

        face_encodings = face_recognition.face_encodings(
            image,
            face_locations
        )

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=0.5
            )

            if True in matches:

                index = matches.index(True)

                return jsonify({
                    "recognized": True,
                    "name": known_names[index]
                })

        return jsonify({
            "recognized": False,
            "message": "Face not recognized"
        })

    except Exception as e:

        print("Recognition error:", e)

        return jsonify({
            "recognized": False,
            "error": str(e)
        }), 500
