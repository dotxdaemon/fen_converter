import os
import json
import csv
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class Shot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    filename = db.Column(db.String(255), nullable=False)
    movie = db.Column(db.String(255))
    director = db.Column(db.String(255))
    dop = db.Column(db.String(255))
    year = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


def migrate_from_files(base_dir: str) -> None:
    """Load data from JSON or CSV files into the database if present."""
    data_file = os.path.join(base_dir, 'data.json')
    csv_file = os.path.join(base_dir, 'data.csv')
    entries = []
    if os.path.exists(data_file):
        try:
            with open(data_file) as f:
                entries.extend(json.load(f))
        except json.JSONDecodeError:
            pass
    if os.path.exists(csv_file):
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)

    if not entries:
        return

    for entry in entries:
        if not entry.get('filename'):
            continue
        shot = Shot(
            title=entry.get('title') or None,
            filename=entry['filename'],
            movie=entry.get('movie') or None,
            director=entry.get('director') or None,
            dop=entry.get('dop') or None,
            year=int(entry['year']) if entry.get('year') not in (None, '', 'null') else None,
        )
        db.session.add(shot)
    db.session.commit()

    if os.path.exists(data_file):
        os.rename(data_file, data_file + '.bak')
    if os.path.exists(csv_file):
        os.rename(csv_file, csv_file + '.bak')
