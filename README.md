# PURE CINEMA

A small Flask application for storing and browsing cinematography reference images.

## Running locally

1. **Create a virtual environment** (optional but recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**

   The app uses OAuth for authentication. Provide the following variables if you
   want to enable GitHub or Google logins:

   - `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`
   - `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`

   You can also set `SECRET_KEY` to override the generated Flask secret and
   `FLASK_DEBUG=1` to enable debug mode.

4. **Run the server**

   ```bash
   python cine_storage/app.py
   ```

The application starts on `http://127.0.0.1:5000/`. Uploaded images are stored
in `cine_storage/uploads` and thumbnails in `cine_storage/thumbs`. A SQLite
database file is created automatically in the repository directory.


