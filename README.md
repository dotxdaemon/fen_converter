# Cinematography Shots Website

A simple Flask application for storing and displaying cinematography shots. Upload images with metadata including title, movie, director, director of photography, and release year. Browse your gallery with all details shown.

Visit `/` to view the gallery. Use `/upload` to add new shots with their metadata.

### Running locally

```bash
pip install -r requirements.txt
python cine_storage/app.py
```

The uploads are stored in `cine_storage/uploads`. 400px thumbnails are generated into `cine_storage/thumbs` on upload. Both directories are ignored by Git via `.gitignore`.
