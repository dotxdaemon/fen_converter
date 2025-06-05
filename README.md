# Cinematography Shots Website

A simple Flask application for storing and displaying cinematography shots. Upload images with metadata including title, movie, director, director of photography, and release year. On upload the application attempts to read EXIF metadata (via `exifread`) and will pre-fill any missing fields automatically. Browse your gallery with all details shown.

Visit `/` to view the gallery. Use `/upload` to add new shots with their metadata.

### Running locally

```bash
pip install -r requirements.txt
python cine_storage/app.py
```

The uploads are stored in `cine_storage/uploads`. They are ignored by Git via `.gitignore`.
