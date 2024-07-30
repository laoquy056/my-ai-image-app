import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import psycopg2
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# API URL and API key for Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="lD811vMk13RcNWRNEBw4"
)

# Heroku DATABASE_URL
DATABASE_URL = os.getenv("postgres://uakqc45e4manqr:pb3aef762040ee7b41b1cfef3104a135ff75e9aea3936cf9ee8b7b956a02145a0@cd5gks8n4kb20g.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d312csom8bnqoo")
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()

UPLOAD_FOLDER = 'uploads'
BASE_OUTPUT_FOLDER = '/Users/hainguyen/Desktop'  # Change the base folder if needed

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        processed BOOLEAN NOT NULL DEFAULT FALSE
    )
''')
conn.commit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_files = request.files.getlist('input_files')
        output_folder = request.form.get('output_folder')
        draw_boxes = 'draw_boxes' in request.form
        show_labels = 'show_labels' in request.form

        if not input_files or not output_folder:
            return "Please select both input files and output folder", 400

        output_folder_path = os.path.join(BASE_OUTPUT_FOLDER, secure_filename(output_folder))

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        for file in input_files:
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                output_path = os.path.join(output_folder_path, filename)

                try:
                    result = CLIENT.infer(file_path, model_id="flycam-thermal/1")
                except Exception as e:
                    app.logger.error(f"Error processing {file_path}: {e}")
                    continue

                image = cv2.imread(file_path)
                has_boxes = False

                if 'predictions' in result and len(result['predictions']) > 0:
                    for prediction in result['predictions']:
                        x_center = int(prediction['x'])
                        y_center = int(prediction['y'])
                        width_box = int(prediction['width'])
                        height_box = int(prediction['height'])

                        x1 = int(x_center - width_box / 2)
                        y1 = int(y_center - height_box / 2)
                        x2 = int(x_center + width_box / 2)
                        y2 = int(y_center + height_box / 2)

                        if draw_boxes:
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            has_boxes = True

                        if show_labels:
                            label = f"{prediction.get('class')}"
                            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            has_boxes = True

                if has_boxes:
                    cv2.imwrite(output_path, image)
                    cursor.execute('INSERT INTO images (filename, processed) VALUES (%s, %s)', (filename, True))
                    conn.commit()
                    app.logger.debug(f"Processed and saved image: {filename}")

        return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
