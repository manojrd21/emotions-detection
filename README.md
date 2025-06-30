# Emotion Detection from Facial Expressions

This project is a deep learning-based application that detects human emotions from facial expressions using a Convolutional Neural Network (CNN). The model is trained on the FER-2013 dataset, and a simple GUI has been developed using Tkinter for real-time prediction.

## Project Structure

```
Emotion-Detection/
├── Dataset/                            # Folder to store the dataset
├── emotions_detection_model.ipynb     # Model training and evaluation notebook
├── gui.py                              # GUI interface for emotion detection
├── haarcascade_frontalface_default.xml # Haar Cascade face detector
├── model_a.json                        # JSON file of model architecture
├── model.weights.h5                    # Trained weights of the model
└── README.md                           # Project documentation (this file)
```

## Example Output

The app detects a face in the uploaded image and predicts the associated emotion, such as:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Details

- Model Type: Convolutional Neural Network (CNN)
- Input Size: 48x48 grayscale image
- Output: One of 7 emotion categories
- Framework: TensorFlow / Keras
- Loss Function: categorical_crossentropy
- Optimizer: Adam

## Dataset

- Name: FER-2013 (Facial Expression Recognition 2013)
- Source: https://www.kaggle.com/datasets/msambare/fer2013
- Labels: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Format: CSV file containing 48x48 pixel grayscale face images

## How to Run

### 1. Clone the Repository

```
git clone https://github.com/manojrd21/emotions-detection.git
cd emotion-detection
```

### 2. Install Dependencies

Required Libraries:

```
tensorflow
numpy
opencv-python
Pillow
```


### 3. Run the GUI

```
python gui.py
```

Then click "Upload Image" to test emotion detection.

## How It Works

1. Load pre-trained model architecture and weights (model_a.json + model.weights.h5)
2. Use OpenCV's Haar Cascade to detect faces
3. Resize the face region to 48x48 and preprocess
4. Predict emotion using the CNN model
5. Display emotion on the GUI

## Limitations

- Trained only on FER-2013 dataset — limited to grayscale 48x48 face images
- Works best with clear, front-facing faces
- May misclassify emotions in poor lighting or side angles

## Future Improvements

- Add webcam support for live detection
- Train on higher-resolution or color datasets
- Improve UI/UX using Tkinter, Streamlit, or Flask
- Add bounding box display for multiple faces

## Author

Manoj Dhanawade  
India | Python, AI/ML Developer
