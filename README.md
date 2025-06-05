# Cinematography Shots Website

A simple Flask application for storing and displaying cinematography shots. Upload images with metadata including title, movie, director, director of photography, and release year. Browse your gallery with all details shown.

Authentication is handled via OAuth (GitHub or Google) using Flask-Login. You must be logged in to upload shots.

Visit `/` to view the gallery. When logged in you can use `/upload` to add new shots with their metadata.

### Running locally

```bash
pip install -r requirements.txt
python cine_storage/app.py
```

Set the following environment variables with your OAuth credentials before running:

- `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- `SECRET_KEY` (optional, for session security)

With these set, start the app and log in via GitHub or Google to upload images.

The uploads are stored in `cine_storage/uploads`. They are ignored by Git via `.gitignore`.
