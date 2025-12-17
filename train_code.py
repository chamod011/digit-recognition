"""
hh.py

Digit recognition training script.

Usage examples:
  python hh.py --epochs 5 --model-out mnist_cnn.h5
  python hh.py --sklearn --model-out digits_mlp.joblib

This script will try to use TensorFlow/Keras (preferred) and fall back to scikit-learn's MLPClassifier
if TensorFlow is not installed or the `--sklearn` flag is provided.
"""

import argparse
import sys
import os
import numpy as np


def train_with_tensorflow(args):
	try:
		import tensorflow as tf
		from tensorflow.keras import layers, models
	except Exception as e:
		print("TensorFlow import failed:", e)
		return False

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0

	x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
	x_test = np.expand_dims(x_test, -1)

	num_classes = 10

	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation="relu"),
		layers.MaxPooling2D((2, 2)),
		layers.Flatten(),
		layers.Dense(128, activation="relu"),
		layers.Dropout(0.5),
		layers.Dense(num_classes, activation="softmax"),
	])

	model.compile(optimizer="adam",
				  loss="sparse_categorical_crossentropy",
				  metrics=["accuracy"])

	model.summary()

	model.fit(x_train, y_train,
			  epochs=args.epochs,
			  batch_size=args.batch_size,
			  validation_split=0.1)

	test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
	print(f"Test accuracy: {test_acc:.4f}")

	# Save model
	out = args.model_out or "mnist_cnn.h5"
	model.save(out)
	print(f"Saved Keras model to {out}")
	return True


def train_with_sklearn(args):
	print("Using scikit-learn MLPClassifier on the classic 'digits' dataset.")
	try:
		from sklearn.datasets import load_digits
		from sklearn.model_selection import train_test_split
		from sklearn.neural_network import MLPClassifier
		from sklearn.metrics import accuracy_score, classification_report
		import joblib
	except Exception as e:
		print("scikit-learn import failed:", e)
		return False

	digits = load_digits()
	X = digits.images  # shape (n_samples, 8, 8)
	y = digits.target
	n_samples = len(X)
	X = X.reshape((n_samples, -1)).astype(float) / 16.0  # normalize

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=args.max_iter, random_state=1)
	clf.fit(X_train, y_train)

	preds = clf.predict(X_test)
	acc = accuracy_score(y_test, preds)
	print(f"Digits dataset test accuracy: {acc:.4f}")
	print(classification_report(y_test, preds))

	out = args.model_out or "digits_mlp.joblib"
	joblib.dump(clf, out)
	print(f"Saved scikit-learn model to {out}")
	return True


def parse_args():
	p = argparse.ArgumentParser(description="Train a digit recognition model (MNIST or sklearn digits).")
	p.add_argument("--sklearn", action="store_true", help="Force use of scikit-learn (fallback).")
	p.add_argument("--epochs", type=int, default=5, help="Number of epochs for TensorFlow training.")
	p.add_argument("--batch-size", type=int, default=128, help="Batch size for TensorFlow training.")
	p.add_argument("--max-iter", type=int, default=200, help="Max iterations for sklearn MLP.")
	p.add_argument("--model-out", type=str, default=None, help="Path to save trained model.")
	return p.parse_args()


def main():
	args = parse_args()

	if args.sklearn:
		ok = train_with_sklearn(args)
		if not ok:
			print("Failed to train with scikit-learn.")
			sys.exit(1)
		return

	# Try TensorFlow first
	ok = train_with_tensorflow(args)
	if not ok:
		print("TensorFlow training unavailable; falling back to scikit-learn.")
		ok2 = train_with_sklearn(args)
		if not ok2:
			print("Both TensorFlow and scikit-learn training failed. Ensure required packages are installed.")
			sys.exit(1)


if __name__ == "__main__":
	main()

