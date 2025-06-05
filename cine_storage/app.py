import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMB_FOLDER'] = THUMB_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + DATABASE_FILE
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    migrate_from_files(BASE_DIR)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'index'

# OAuth setup
oauth = OAuth(app)
oauth.register(
    name='github',
    client_id=os.environ.get('GITHUB_CLIENT_ID'),
    client_secret=os.environ.get('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)
oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMB_FOLDER, exist_ok=True)


class User(UserMixin):
    def __init__(self, user_id: str, name: str):
        self.id = user_id
        self.name = name


users = {}


@login_manager.user_loader
def load_user(user_id: str):
    return users.get(user_id)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






def update_indices(entry):
    for field, index in search_indices.items():
        value = str(entry.get(field, "")).lower()
        index.setdefault(value, []).insert(0, entry)


@app.route('/login/<provider>')
def oauth_login(provider: str):
    client = oauth.create_client(provider)
    redirect_uri = url_for('authorize', provider=provider, _external=True)
    return client.authorize_redirect(redirect_uri)


@app.route('/authorize/<provider>')
def authorize(provider: str):
    client = oauth.create_client(provider)
    token = client.authorize_access_token()
    if provider == 'github':
        resp = client.get('user', token=token)
        info = resp.json()
        uid = str(info['id'])
        name = info.get('name') or info.get('login')
    else:
        resp = client.get('userinfo', token=token)
        info = resp.json()
        uid = info['id']
        name = info.get('name')
    user = User(uid, name)
    users[uid] = user
    login_user(user)
    return redirect(url_for('index'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/')
def index():


@app.route('/upload', methods=['GET', 'POST'])
@login_required
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

    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/thumbs/<filename>')
def thumbnail(filename):
    return send_from_directory(app.config['THUMB_FOLDER'], filename)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(debug=debug)
