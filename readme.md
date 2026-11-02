
# Indian Cricketer Face Recognition System

## Project Overview

This project is an image classification application that identifies Indian cricketers using a Convolutional Neural Network (CNN).

The trained deep learning model is integrated with a Flask web application where users can upload an image and get the predicted cricketer's name along with the confidence score.

## Technologies Used

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Flask
- HTML
- CSS
- NumPy
- Pillow

## Recognized Cricketers

The model can recognize the following 11 Indian cricketers:

1. Hardik Pandya
2. Jasprit Bumrah
3. KL Rahul
4. Mohammed Siraj
5. Ravindra Jadeja
6. Rishabh Pant
7. Rohit Sharma
8. Shubman Gill
9. Suryakumar Yadav
10. Venkatesh Iyer
11. Yuzvendra Chahal

## Project Structure

```text
Cricketer_pic_detection_project/
│
├── app.py
├── cricketer_pic_detection_model.keras
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Installation

1. Clone the repository:

```bash
git https://github.com/ArmanAli1234/Cricketer_pic_detection_model.git
```

2. Navigate to the project directory:

```bash
cd Cricketer_pic_detection_project
```

3. Install the required libraries:

```bash
pip install flask tensorflow numpy pillow or 
pip install -r requirements.txt
```

## Run the Application

Execute the following command:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## How It Works

1. Upload an image of an Indian cricketer.
2. The Flask application receives the uploaded image.
3. The image is resized and preprocessed.
4. The trained CNN model predicts the cricketer.
5. The application displays the predicted name and confidence score.

## Model

The project uses a trained CNN model saved in `.keras` format.

**Model File:** `cricketer_pic_detection_model.keras`

## Future Improvements

- Improve model accuracy.
- Add prediction history.
- Display the top 3 predictions.
- Improve the user interface.

## Author

Developed as a Deep Learning project using Python and Flask.
