# Face Recognition using PCA, LDA, and MLP

This Python script demonstrates face recognition using Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and a Multi-Layer Perceptron (MLP) Classifier. The script loads a dataset of faces, performs dimensionality reduction with PCA, applies LDA for further feature extraction, and finally trains an MLP Classifier for face recognition.

## Dependencies

Make sure you have the following Python libraries installed:

- Matplotlib
- NumPy
- OpenCV
- scikit-learn

You can install them using:

```bash
pip install matplotlib numpy opencv-python scikit-learn
```

## Dataset

The script uses a dataset of faces located in the "datasets/faces/" directory. Each subdirectory corresponds to a different person, and images inside these directories are used for training and testing.

## Running the Script

Execute the script in a Python environment:

```bash
python face_recognition.py
```

## Output

1. Eigenfaces: The script displays eigenfaces extracted from the PCA analysis.
2. MLP Training: It prints information about the MLP Classifier, including the model weights.
3. Face Recognition Results: The script evaluates the accuracy of the face recognition model and displays a gallery of test images with predicted and true labels.

## Note

- The script uses PCA for dimensionality reduction and LDA for feature extraction before training the MLP Classifier.
- Ensure the dataset directory structure follows the format mentioned above.

Feel free to customize the script and adapt it to your specific use case. If you encounter any issues or have suggestions for improvement, please create an issue or submit a pull request.

Happy coding!