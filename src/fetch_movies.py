"""Descarga un corpus de peliculas desde TMDB y lo guarda en data/movies.json.

Se corre una sola vez. El JSON resultante se versiona en el repo para que el
resto del pipeline sea reproducible sin necesitar la API key.

    python -m src.fetch_movies --target 800
"""

import argparse
import json
import sys
import time

import requests

from src.config import DATA_DIR, MOVIES_JSON, TMDB_API_KEY, TMDB_BASE_URL

RESULTS_PER_PAGE = 20


def _get(path: str, **params) -> dict:
    params["api_key"] = TMDB_API_KEY
    response = requests.get(f"{TMDB_BASE_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_genre_map(language: str) -> dict[int, str]:
    """TMDB devuelve ids de genero; necesitamos sus nombres."""
    payload = _get("/genre/movie/list", language=language)
    return {genre["id"]: genre["name"] for genre in payload["genres"]}


def fetch_movies(target: int, language: str, min_votes: int) -> list[dict]:
    genres = fetch_genre_map(language)
    movies: list[dict] = []
    seen: set[int] = set()
    page = 1

    while len(movies) < target and page <= 500:
        payload = _get(
            "/discover/movie",
            language=language,
            sort_by="popularity.desc",
            include_adult="false",
            page=page,
            **{"vote_count.gte": min_votes},
        )

        for item in payload["results"]:
            overview = (item.get("overview") or "").strip()
            # Sin resumen la pelicula no sirve: es justamente lo que vectorizamos.
            if not overview or item["id"] in seen:
                continue
            seen.add(item["id"])
            movies.append(
                {
                    "id": item["id"],
                    "title": item.get("title", ""),
                    "original_title": item.get("original_title", ""),
                    "year": (item.get("release_date") or "")[:4],
                    "genres": [genres.get(g, str(g)) for g in item.get("genre_ids", [])],
                    "overview": overview,
                    "vote_average": item.get("vote_average"),
                    "poster_path": item.get("poster_path"),
                }
            )
            if len(movies) >= target:
                break

        print(f"  pagina {page}: {len(movies)}/{target} peliculas", file=sys.stderr)
        if page >= payload.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.25)  # cortesia con la API

    return movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga peliculas de TMDB.")
    parser.add_argument("--target", type=int, default=800, help="cantidad de peliculas")
    parser.add_argument("--language", default="es-ES", help="idioma de los resumenes")
    parser.add_argument("--min-votes", type=int, default=100, help="votos minimos en TMDB")
    args = parser.parse_args()

    if not TMDB_API_KEY:
        sys.exit(
            "Falta TMDB_API_KEY.\n"
            "  1. Sacá una clave gratis en https://www.themoviedb.org/settings/api\n"
            "  2. cp .env.example .env\n"
            "  3. Pegá la clave en .env"
        )

    print(f"Descargando {args.target} peliculas en {args.language}...", file=sys.stderr)
    movies = fetch_movies(args.target, args.language, args.min_votes)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MOVIES_JSON.write_text(
        json.dumps(movies, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Guardadas {len(movies)} peliculas en {MOVIES_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
