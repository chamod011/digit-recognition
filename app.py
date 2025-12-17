#!/usr/bin/env python3
"""Flask server to accept canvas drawings and return digit predictions.

Run:
  python app.py --model mnist_cnn.h5 --port 5000

Then open `web_app.html` in a browser (http://localhost:5000/web_app.html) or open the file directly and point to the server.
"""

import argparse
import base64
import io
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np


def load_model_auto(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.h5', '.keras'):
        try:
            from tensorflow.keras.models import load_model
            return 'keras', load_model(path)
        except Exception as e:
            raise RuntimeError('Failed to load Keras model: ' + str(e))
    if ext in ('.joblib', '.pkl'):
        try:
            import joblib
            return 'sklearn', joblib.load(path)
        except Exception as e:
            raise RuntimeError('Failed to load sklearn model: ' + str(e))
    # fallback
    try:
        from tensorflow.keras.models import load_model
        return 'keras', load_model(path)
    except Exception:
        import joblib
        return 'sklearn', joblib.load(path)


def preprocess_for_keras_pil(img_pil, invert=False):
    img = img_pil.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype('float32')
    if invert:
        arr = 255.0 - arr
    arr = arr / 255.0
    arr = arr.reshape((1, 28, 28, 1))
    return arr


def preprocess_for_sklearn_pil(img_pil, invert=False):
    img = img_pil.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.array(img).astype('float32')
    if invert:
        arr = 255.0 - arr
    arr = (arr / 255.0) * 16.0
    arr = arr.reshape((1, -1))
    return arr


def create_app(model_path):
    app = Flask(__name__, static_folder='.')
    CORS(app)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    model_type, model = load_model_auto(model_path)
    print('Loaded model type:', model_type)


    @app.route('/')
    def index():
        return send_from_directory('.', 'web_app.html')


    @app.route('/web_app.html')
    def web_app():
        return send_from_directory('.', 'web_app.html')


    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        img_b64 = data.get('image')
        invert = bool(data.get('invert', False))

        if not img_b64:
            return jsonify({'error': 'No image provided'}), 400

        # handle data URL
        if img_b64.startswith('data:'):
            img_b64 = img_b64.split(',', 1)[1]

        try:
            decoded = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(decoded))
        except Exception as e:
            return jsonify({'error': 'Failed to decode image: ' + str(e)}), 400

        try:
            if model_type == 'keras':
                x = preprocess_for_keras_pil(img, invert=invert)
                preds = model.predict(x)
                probs = preds[0].astype(float).tolist()
                label = int(np.argmax(preds, axis=1)[0])
            else:
                x = preprocess_for_sklearn_pil(img, invert=invert)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(x)[0].astype(float).tolist()
                    label = int(np.argmax(probs))
                else:
                    label = int(model.predict(x)[0])
                    probs = None
        except Exception as e:
            return jsonify({'error': 'Prediction failed: ' + str(e)}), 500

        return jsonify({'label': label, 'probs': probs, 'model_type': model_type})

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='mnist_cnn.h5', help='Path to model file (.h5 or .joblib)')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', default=5000, type=int)
    args = p.parse_args()

    app = create_app(args.model)
    print(f'Serving on http://{args.host}:{args.port}  (model={args.model})')
    app.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
