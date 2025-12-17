# Digit Recognition

This repository contains a small example project for training and testing digit recognition models. It supports two training flows:

- A TensorFlow / Keras convolutional neural network (CNN) trained on MNIST (preferred).
- A scikit-learn `MLPClassifier` trained on the classic `digits` dataset (lightweight fallback).

Files of interest
- `hh.py` — CLI script to train models. Tries Keras first, falls back to scikit-learn or can be forced with `--sklearn`.
- `mnist_cnn.h5` — (example) Keras model file (if present in repo root).
- `test_model.ipynb` — Notebook to load a saved model and run predictions on your own images.
- `requirements.txt` — Minimal recommended packages. TensorFlow is optional and large; install it only if you want the CNN flow.

Quick setup (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the lightweight requirements (sklearn flow):

```powershell
pip install -r requirements.txt
```

3. (Optional) Install TensorFlow if you want to train the Keras CNN:

```powershell
pip install tensorflow        # or: pip install tensorflow-cpu
```

Training

- Train using TensorFlow (if installed):

```powershell
python hh.py --epochs 5 --batch-size 128 --model-out mnist_cnn.h5
```

- Force scikit-learn training (lightweight):

```powershell
python hh.py --sklearn --max-iter 300 --model-out digits_mlp.joblib
```

Model outputs

- By default the Keras trainer saves `mnist_cnn.h5` (HDF5 Keras model). The sklearn trainer saves `digits_mlp.joblib`.
- You can override the output path with `--model-out <path>`.

Testing your own images with the notebook

1. Open `test_model.ipynb` in Jupyter or VS Code Notebook.
2. Update the `MODEL_PATH` variable to point to your saved model (for example `mnist_cnn.h5` or `digits_mlp.joblib`).
3. Update `IMAGE_PATH` to point to the image you want to test (example: `test_digit.png`).
4. If your image uses white background with a dark digit (or vice versa), set `INVERT = True` in the notebook.
5. Run the notebook cells in order — the notebook contains preprocessing helpers for both Keras (28x28) and sklearn (8x8) models and displays the predicted label.

Tips for good test images

- Crop tightly around the digit so the digit fills most of the frame.
- Use a plain background and high contrast between digit and background.
- Save images in grayscale if possible; the notebook will convert color images to grayscale.
- For Keras MNIST models: images should be similar to MNIST style (28x28, centered). The notebook resizes automatically but good input helps accuracy.

Next steps / suggestions

- Add early stopping, model checkpoints, or data augmentation to `hh.py` for improved training.
- Add a small set of example test images in a `samples/` folder to make quick verification easier.
- If you want, I can create a sample `test_digit.png` and run the notebook here to show a prediction.

Web drawing UI (draw on canvas and predict)

This repository also includes a simple HTML canvas client (`web_app.html`) and a small Flask server (`app.py`) so you can draw digits in your browser and get live predictions from your model.

1. Start the Flask server from the repository root (ensure your venv is activated and dependencies installed):

```powershell
python app.py --model mnist_cnn.h5 --port 5000
```

If your model is named differently, replace `mnist_cnn.h5` with your model path (supports `.h5` Keras models and `.joblib` scikit-learn models).

2. Open the client in your browser:

- Visit `http://localhost:5000/web_app.html` to open the drawing canvas. Draw with mouse or touch, then click **Predict**.
- Use the **Invert** checkbox if your canvas output has reversed colors (white digit on black vs black digit on white).

How it works
- The client captures the canvas as a PNG and POSTs it to `/predict` on the server.
- The server decodes the PNG, preprocesses into the correct shape (28x28 for Keras MNIST models, 8x8 for the sklearn digits model), and returns a JSON response `{ label, probs, model_type }`.

Troubleshooting for the web UI
- If you get `Model file not found`, double-check the `--model` path you passed to `app.py` and that the server is running in the same folder as the model.
- If predictions are consistently wrong, try the **Invert** checkbox and draw a larger, centered digit.
- If the server returns an error about TensorFlow imports but you only intend to use sklearn, train and supply a `.joblib` model instead.

Next actions I can do for you
- Add automatic centering and normalization in the server to better match MNIST preprocessing.
- Add a `samples/` folder with a few example images and a test script.
- Create a small Dockerfile to run the server consistently.

License

This repository is provided as a simple example — add a license file if you plan to share or publish broadly.

If you want any part of this README expanded (more troubleshooting, examples, or CI instructions), tell me which sections to expand.