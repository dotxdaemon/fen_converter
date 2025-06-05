import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import exifread

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FILE = os.path.join(BASE_DIR, 'data.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_entries():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []


def save_entries(entries) -> None:
    with open(DATA_FILE, 'w') as f:
        json.dump(entries, f)


def read_exif_metadata(file_obj):
    """Extract select EXIF tags as strings."""
    metadata = {}
    try:
        file_obj.seek(0)
        tags = exifread.process_file(file_obj, details=False)
    except Exception:
        return metadata

    def tag_value(name):
        val = tags.get(name)
        return str(val) if val else ''

    metadata['title'] = tag_value('Image XPTitle') or tag_value('Image ImageDescription')
    metadata['movie'] = tag_value('Image XPSubject')
    metadata['director'] = tag_value('Image XPAuthor')
    metadata['dop'] = tag_value('Image Artist')
    dt = tag_value('EXIF DateTimeOriginal')
    if dt:
        metadata['year'] = dt.split(':')[0]
    return metadata


@app.route('/')
def index():
    entries = load_entries()
    return render_template('index.html', entries=entries)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    file = request.files.get('shot')
    title = request.form.get('title', '')
    movie = request.form.get('movie', '')
    director = request.form.get('director', '')
    dop = request.form.get('dop', '')
    year = request.form.get('year', '')
    if file and allowed_file(file.filename):
        exif = read_exif_metadata(file.stream)
        title = title or exif.get('title', '')
        movie = movie or exif.get('movie', '')
        director = director or exif.get('director', '')
        dop = dop or exif.get('dop', '')
        year = year or exif.get('year', '')
        file.stream.seek(0)
        filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        entries = load_entries()
        entries.insert(0, {
            'title': title,
            'filename': filename,
            'movie': movie,
            'director': director,
            'dop': dop,
            'year': year,
        })
        save_entries(entries)
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(debug=debug)
