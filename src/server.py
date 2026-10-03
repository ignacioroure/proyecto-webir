"""Servidor web: expone el buscador como API y sirve la interfaz grafica.

    python -m src.server                 # http://127.0.0.1:8000
    python -m src.server --port 8080

Los indices y el modelo de embeddings se cargan una sola vez al arrancar, asi
cada consulta solo paga vectorizar la consulta y rankear el corpus.
"""

import argparse
import time
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles

from src.config import TMDB_POSTER_URL, WEB_DIR
from src.corpus import load_movies
from src.search import EmbeddingSearcher, TfidfSearcher


def load_state(app: FastAPI) -> None:
    """Carga corpus, indices y modelo en el estado de la aplicacion."""
    movies = load_movies()
    app.state.movies = movies
    app.state.searchers = {
        "tfidf": TfidfSearcher(movies),
        "embeddings": EmbeddingSearcher(movies),
    }
    # El modelo se carga perezosamente: lo forzamos aca para que la primera
    # consulta no tarde varios segundos.
    app.state.searchers["embeddings"].model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # main() ya cargo todo antes de arrancar uvicorn, para que un indice
    # faltante corte con su mensaje y no con un traceback desde aca adentro.
    # Esto cubre el caso de levantar la app con `uvicorn src.server:app`.
    if not hasattr(app.state, "searchers"):
        load_state(app)
    yield


app = FastAPI(title="Buscador semantico de peliculas", lifespan=lifespan)


def movie_payload(movie: dict, score: float) -> dict:
    poster = movie.get("poster_path")
    return {
        "title": movie["title"],
        "year": movie.get("year") or None,
        "genres": movie.get("genres", []),
        "overview": movie["overview"],
        "vote_average": movie.get("vote_average"),
        "poster_url": f"{TMDB_POSTER_URL}{poster}" if poster else None,
        "score": score,
    }


def decade_of(movie: dict) -> int | None:
    year = movie.get("year") or ""
    return int(year) // 10 * 10 if year.isdigit() else None


def filter_mask(movies: list[dict], genre: str | None, decade: int | None) -> np.ndarray | None:
    """Mascara de las peliculas que pasan los filtros; None si no hay filtros."""
    if not genre and decade is None:
        return None
    return np.array(
        [
            (not genre or genre in movie.get("genres", []))
            and (decade is None or decade_of(movie) == decade)
            for movie in movies
        ],
        dtype=bool,
    )


@app.get("/api/info")
def info(request: Request) -> dict:
    movies = request.app.state.movies
    searchers = request.app.state.searchers
    genres = Counter(genre for movie in movies for genre in movie.get("genres", []))
    decades = Counter(decade_of(movie) for movie in movies)
    decades.pop(None, None)
    return {
        "movies": len(movies),
        "tfidf_dims": searchers["tfidf"].matrix.shape[1],
        "embedding_dims": searchers["embeddings"].matrix.shape[1],
        "embedding_model": searchers["embeddings"].model_name,
        "genres": [{"name": name, "count": n} for name, n in sorted(genres.items())],
        "decades": [
            {"decade": decade, "count": n} for decade, n in sorted(decades.items(), reverse=True)
        ],
    }


@app.get("/api/search")
def search(
    request: Request,
    q: str = Query(min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=50),
    genre: str | None = Query(None, max_length=50),
    decade: int | None = None,
) -> dict:
    movies = request.app.state.movies
    allowed = filter_mask(movies, genre, decade)
    methods = {}
    for key, searcher in request.app.state.searchers.items():
        start = time.perf_counter()
        results = searcher.search(q, k, allowed)
        elapsed_ms = (time.perf_counter() - start) * 1000
        methods[key] = {
            "ms": round(elapsed_ms, 1),
            "results": [movie_payload(movies[index], score) for index, score in results],
        }
    candidates = len(movies) if allowed is None else int(allowed.sum())
    return {"query": q, "candidates": candidates, "methods": methods}


# Va despues de las rutas de la API: montado en "/" atrapa todo lo demas.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    parser = argparse.ArgumentParser(description="Levanta la interfaz web del buscador.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Antes de arrancar el servidor: si falta un indice, sys.exit muestra su
    # mensaje limpio en vez de quedar envuelto en el traceback del lifespan.
    load_state(app)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
