"""Rutas y constantes compartidas por todo el proyecto."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

MOVIES_JSON = DATA_DIR / "movies.json"
SAMPLE_MOVIES_JSON = DATA_DIR / "sample_movies.json"
TFIDF_INDEX = INDEX_DIR / "tfidf.pkl"
EMBEDDINGS_INDEX = INDEX_DIR / "embeddings.npz"

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Modelo multilingue: mapea espanol e ingles al mismo espacio vectorial,
# asi la consulta puede estar en un idioma y el resumen en otro.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
