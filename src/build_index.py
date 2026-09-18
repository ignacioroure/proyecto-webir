"""Construye los dos indices vectoriales sobre los resumenes.

    python -m src.build_index            # ambos
    python -m src.build_index --method tfidf

Los artefactos van a index/ y no se versionan: se regeneran en segundos.
"""

import argparse
import pickle
import sys

import numpy as np

from src.config import EMBEDDING_MODEL, EMBEDDINGS_INDEX, INDEX_DIR, TFIDF_INDEX
from src.corpus import document_text, load_movies
from src.stopwords import SPANISH_STOPWORDS


def build_tfidf(documents: list[str]) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words=SPANISH_STOPWORDS,
        # min_df=2 descarta terminos que aparecen en un solo documento (ruido,
        # nombres propios irrepetibles); max_df=0.5 descarta los demasiado comunes.
        min_df=2,
        max_df=0.5,
        ngram_range=(1, 2),
        # Suaviza el peso de la frecuencia: 1+log(tf) en vez de tf crudo.
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(documents)
    # TfidfVectorizer ya normaliza cada fila con L2, asi que el producto
    # punto entre filas es directamente la similitud coseno.

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with TFIDF_INDEX.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, handle)

    print(
        f"TF-IDF: {matrix.shape[0]} documentos x {matrix.shape[1]} terminos "
        f"({matrix.nnz} valores no nulos, densidad "
        f"{matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%})",
        file=sys.stderr,
    )


def build_embeddings(documents: list[str]) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Cargando modelo {EMBEDDING_MODEL}...", file=sys.stderr)
    model = SentenceTransformer(EMBEDDING_MODEL)
    matrix = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        # Normalizados a norma 1: el producto punto es la similitud coseno.
        normalize_embeddings=True,
    ).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_INDEX, matrix=matrix, model=EMBEDDING_MODEL)

    print(
        f"Embeddings: {matrix.shape[0]} documentos x {matrix.shape[1]} dimensiones",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye los indices vectoriales.")
    parser.add_argument(
        "--method", choices=["tfidf", "embeddings", "both"], default="both"
    )
    args = parser.parse_args()

    movies = load_movies()
    documents = [document_text(movie) for movie in movies]
    print(f"Corpus: {len(documents)} peliculas", file=sys.stderr)

    if args.method in ("tfidf", "both"):
        build_tfidf(documents)
    if args.method in ("embeddings", "both"):
        build_embeddings(documents)


if __name__ == "__main__":
    main()
