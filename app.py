
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained CNN model
model = load_model("cricketer_pic_detection_model.keras")

# Image dimensions used during training
IMG_SIZE = (224, 224)

# All 11 Cricketer Classes
# IMPORTANT: Keep training dataset class order exactly.
class_names = [
    "Hardik Pandya",
    "Jasprit Bumrah",
    "KL Rahul",
    "Mohd Siraj",
    "Ravindra Jadeja",
    "Rishabh Pant",
    "Rohit Sharma",
    "Shubman Gill",
    "Suryakumar Yadav",
    "Venkatesh Iyer",
    "Yuzvendra Chahal"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please upload a cricketer image."

        else:
            try:

                # Read uploaded image
                image = Image.open(file.stream).convert("RGB")

                # Resize image
                image = image.resize(IMG_SIZE)

                # Convert image into array
                image = np.array(image, dtype=np.float32)

                # Normalize image
                # Use only if training used rescale=1./255
                image = image / 255.0

                # Add batch dimension
                image = np.expand_dims(image, axis=0)

                # Model prediction
                result = model.predict(image, verbose=0)

                # Get predicted class index
                predicted_index = int(np.argmax(result[0]))

                # Get cricketer name
                prediction = class_names[predicted_index]

                # Calculate confidence
                confidence = round(
                    float(result[0][predicted_index]) * 100, 2
                )

            except Exception as e:
                error = f"Something went wrong: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
