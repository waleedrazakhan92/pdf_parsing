from utils.processing_pipeline import *
from utils.common_functions import make_folder
import os
from flask import Flask, render_template, request, flash, jsonify,send_file
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename

app = Flask(__name__) #app name
run_with_ngrok(app)

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'downloads/'
ALLOWED_EXTENSIONS = {'pdf'}

make_folder(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return render_template('home_direct.html')


        all_documents = []
        all_uploaded_files = request.files.getlist('file')
        for pdf_file in all_uploaded_files:
            if pdf_file and allowed_file(pdf_file.filename):
                filename = secure_filename(pdf_file.filename)
                pdf_path = os.path.join(UPLOAD_FOLDER, filename)
                pdf_file.save(pdf_path)
                all_documents.append(pdf_path)


        save_junk = False
        save_images = False
        save_zips = False
        display_info=False
        op_orientation = True

        path_write_all = 'downloads/processed_pdfs_flask/'
        if os.path.isdir(path_write_all):  shutil.rmtree(path_write_all)
        process_all_documents(path_write_all,all_documents,save_junk=save_junk,save_images=save_images,save_zips=save_zips,op_orientation=op_orientation,display_info=display_info)

        # Provide link to download the generated zip file
        zip_url = f'downloads/all_zips.zip'  # You'll define this route later
        all_pdfs_path = os.path.join(path_write_all,'pdfs/')
        shutil.make_archive(zip_url.split('.zip')[0], 'zip', all_pdfs_path)

        return render_template('download.html', zip_url=zip_url)  # Render the template

    return render_template('home_direct.html')

@app.route('/downloads/<filename>', methods=['GET'])
def download_file(filename):
    zip_file_path = f'downloads/{filename}'
    return send_file(zip_file_path, as_attachment=True)



if __name__ == "__main__":
    app.run()

