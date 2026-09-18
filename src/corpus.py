"""Carga del corpus y construccion del texto que se vectoriza."""

import json
import sys

from src.config import MOVIES_JSON, SAMPLE_MOVIES_JSON


def load_movies() -> list[dict]:
    """Devuelve el corpus real si existe; si no, el de demostracion.

    El fallback permite que cualquiera clone el repo y corra el buscador sin
    tener una API key de TMDB.
    """
    if MOVIES_JSON.exists():
        return json.loads(MOVIES_JSON.read_text(encoding="utf-8"))

    if SAMPLE_MOVIES_JSON.exists():
        print(
            f"[aviso] usando el corpus de demostracion ({SAMPLE_MOVIES_JSON.name}, "
            "12 peliculas).\n"
            "        Para el corpus real: python -m src.fetch_movies --target 800",
            file=sys.stderr,
        )
        return json.loads(SAMPLE_MOVIES_JSON.read_text(encoding="utf-8"))

    sys.exit(f"No hay corpus. Corré: python -m src.fetch_movies --target 800")


def document_text(movie: dict) -> str:
    """Texto indexable de una pelicula.

    Concatenamos titulo y generos al resumen: son senales cortas pero muy
    informativas, y hacen que una consulta como "terror" matchee aunque el
    resumen nunca use esa palabra.
    """
    parts = [movie["title"], " ".join(movie.get("genres", [])), movie["overview"]]
    return " ".join(part for part in parts if part)
