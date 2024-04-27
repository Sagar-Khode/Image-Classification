from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from image_utils import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained Random Forest model
model_path = 'Deployment/models/best_random_forest_modelx_new.pkl'
model = load_model(model_path)

decoder_path = 'Deployment/models/label_encoder_newx.pkl'
label_encoder = load_decoder(decoder_path)


@app.route('/', methods=['GET', 'POST'])

def upload_image():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file temporarily
            uploaded_file.save('temp_image.jpg')

        img_array = preprocess_image('temp_image.jpg')
        prediction = predict_image(model,label_encoder, img_array)

        # Process prediction and return result
        result = "Predicted Class: {}".format(prediction)

        return render_template('index.html', result=result)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)


