import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATABASE_FILE = os.path.join(BASE_DIR, 'shots.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    migrate_from_files(BASE_DIR)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




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
    entries = Shot.query.order_by(Shot.id.desc()).all()
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
        shot = Shot(
            title=title or None,
            filename=filename,
            movie=movie or None,
            director=director or None,
            dop=dop or None,
            year=int(year) if year else None,
        )
        db.session.add(shot)
        db.session.commit()
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(debug=debug)
