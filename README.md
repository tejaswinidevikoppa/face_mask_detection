# Face Mask Detection

This project is a face mask detection application using Convolutional Neural Networks (CNNs) implemented with TensorFlow and Keras. It utilizes a web interface built with Flask to interact with a real-time webcam feed for mask detection. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Interface](#web-interface)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project detects whether a person is wearing a mask or not using a trained CNN model. The application leverages OpenCV for real-time image processing and TensorFlow/Keras for building and deploying the machine learning model. A Flask web application serves as the frontend interface.

## Features

- Real-time mask detection using a webcam.
- Displays detection results with bounding boxes and labels.
- Web interface to view the webcam feed and detection results.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/face-mask-detection.git
    cd face-mask-detection
    ```

2. **Install required packages:**

    Make sure you have Python3 installed. Then, install the necessary packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    Create a `requirements.txt` file with the following content:

    ```
    numpy
    opencv-python
    tensorflow
    keras
    flask
    ```

3. **Download Pre-trained Models and Haar Cascade:**

    - Place the `model-{epoch:03d}.model` file (from training) in the project directory.
    - Download `haarcascade_frontalface_default.xml` from the [OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the project directory.

## Usage

1. **Start the Flask server:**

    ```bash
    python app.py
    ```

    This will start a local server on `http://127.0.0.1:5000`.

2. **Access the Web Interface:**

    Open your web browser and navigate to `http://127.0.0.1:5000/cam` to view the webcam feed and mask detection results.

## Model Training

1. **Prepare Data:**

    - Place your dataset in a folder named `dataset` in the project directory.
    - The dataset should be organized in subfolders, each representing a different category (e.g., `MASK` and `NO_MASK`).

2. **Train the Model:**

    Run the script `train_model.py` to train the model:

    ```bash
    python train_model.py
    ```

    This will generate model checkpoints saved as `model-{epoch:03d}.model`.

## Web Interface

The Flask web application serves as the user interface. The `app.py` script defines the routes and handles webcam interactions.

- `/cam`: Displays the webcam feed.
- `/mask`: Runs the mask detection algorithm on the webcam feed.

## File Structure

```
face-mask-detection/
│
├── dataset/                      # Folder containing dataset
│   ├── MASK/
│   └── NO_MASK/
│
├── model-{epoch:03d}.model       # Trained model checkpoint(s)
├── haarcascade_frontalface_default.xml # Haar cascade file for face detection
├── app.py                        # Flask application script
├── train_model.py                # Script for training the model
├── requirements.txt              # Python packages required
└── README.md                     # This README file
```

## Contributing

If you have suggestions for improvements or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
