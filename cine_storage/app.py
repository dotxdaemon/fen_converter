import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

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


def build_indices(entries):
    indices = {"title": {}, "movie": {}, "dop": {}, "year": {}}
    for entry in entries:
        for field in indices.keys():
            value = str(entry.get(field, "")).lower()
            indices[field].setdefault(value, []).append(entry)
    return indices


# Load existing entries and build initial indices
entries = load_entries()
search_indices = build_indices(entries)


def save_entries(entries) -> None:
    with open(DATA_FILE, 'w') as f:
        json.dump(entries, f)


def update_indices(entry):
    for field, index in search_indices.items():
        value = str(entry.get(field, "")).lower()
        index.setdefault(value, []).insert(0, entry)


@app.route('/')
def index():
    return render_template('index.html', entries=entries, query='')


@app.route('/search')
def search():
    q = request.args.get('q', '').strip().lower()
    results = []
    seen = set()
    if q:
        for field_index in search_indices.values():
            for key, items in field_index.items():
                if q in key:
                    for entry in items:
                        fid = entry.get('filename')
                        if fid not in seen:
                            results.append(entry)
                            seen.add(fid)
    return render_template('index.html', entries=results, query=q)

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
        filename = datetime.now().strftime('%Y%m%d%H%M%S_') + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        entry = {
            'title': title,
            'filename': filename,
            'movie': movie,
            'director': director,
            'dop': dop,
            'year': year,
        }
        entries.insert(0, entry)
        update_indices(entry)
        save_entries(entries)
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(debug=debug)
