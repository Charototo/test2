# 导入必要的模块
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import io
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Define data augmentation generator
image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

# Instantiate a new Flask web application
app = Flask(__name__)

# Load a pre-trained deep convolutional neural network model using TensorFlow's load_model method
model = tf.keras.models.load_model("D:\Coding\GCPapp\my_model.keras")


# Define the route
@app.route('/', methods=['GET', 'POST'])
def index():
    # Check if the current request is POST method
    if request.method == 'POST':
        # Retrieve the uploaded file from the request named 'file'
        file = request.files['file']

        # Check if the file exists
        if file:
            # Read the file content and convert it to a BytesIO object
            bytes_io = io.BytesIO(file.read())

            # Load the image from BytesIO object and resize it to 180x180 using TensorFlow's load_img function
            image = tf.keras.preprocessing.image.load_img(bytes_io, target_size=(180, 180))

            # Convert the image object to a numpy array
            data = tf.keras.preprocessing.image.img_to_array(image)

            # Expand the dimensions to make it a 4D array as ImageDataGenerator.flow() requires batch-like data
            data = np.expand_dims(data, axis=0)

            # Perform data augmentation on the image using the defined image_generator
            ig = image_generator.flow(data)

            # Predict the results using the loaded model
            predictions = model.predict(ig[0])

            result_70 = "Non-Pneumonia" if predictions[0][0] <= 0.7 else "Pneumonia"
            result_50 = "Non-Pneumonia" if predictions[0][0] <= 0.5 else "Pneumonia"

            return f'''
            <div style="font-size:24px; color:blue; text-align:center; margin-top:200px;">
                <p>Threshold 0.7 Prediction: {result_70}, Confidence Score: {predictions[0][0]}</p>
                <p>Threshold 0.5 Prediction: {result_50}, Confidence Score: {predictions[0][0]}</p>
            </div>
            '''
        else:
            # Return an error message if no file is uploaded or the file is invalid
            return '<div style="font-size:24px; color:red; text-align:center; margin-top:200px;">Invalid file format</div>'

    # Render and return the upload.html template for GET requests
    return render_template('upload.html')


# Run the Flask application if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
