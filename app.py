# pip install Flask tensorflow pillow numpy

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('model/best_retinopathy_model.h5')  

def predict_image(filepath):
    img = Image.open(filepath).convert('RGB')
    w, h = img.size
    min_dim = min(w, h)
    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                    (w + min_dim) // 2, (h + min_dim) // 2))
    img = img.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = 'No Diabetic Retinopathy' if prediction > 0.5 else 'Diabetic Retinopathy'
    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
