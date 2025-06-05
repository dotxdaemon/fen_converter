import os
import json
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

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATA_FILE = os.path.join(BASE_DIR, 'data.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

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


def load_entries():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []


def save_entries(entries) -> None:
    with open(DATA_FILE, 'w') as f:
        json.dump(entries, f)


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
    entries = load_entries()
    return render_template('index.html', entries=entries)

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
