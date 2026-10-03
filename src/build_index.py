"""Construye los dos indices vectoriales sobre los resumenes.

    python -m src.build_index            # ambos
    python -m src.build_index --method tfidf

Los artefactos van a index/ y no se versionan: se regeneran en segundos.
Cada indice guarda los ids de las peliculas que contiene, para que el buscador
detecte si el corpus cambio despues de indexar.
"""

import argparse
import pickle
import re
import sys

import numpy as np

from src.config import EMBEDDING_MODEL, EMBEDDINGS_INDEX, INDEX_DIR, TFIDF_INDEX
from src.corpus import document_text, load_movies
from src.stopwords import SPANISH_STOPWORDS


def build_tfidf(documents: list[str], ids: list[int]) -> None:
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
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "ids": ids}, handle)

    print(
        f"TF-IDF: {matrix.shape[0]} documentos x {matrix.shape[1]} terminos "
        f"({matrix.nnz} valores no nulos, densidad "
        f"{matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%})",
        file=sys.stderr,
    )


def count_tokens(tokenizer, text: str) -> int:
    # verbose=False: el tokenizer avisa cuando un texto supera el limite del
    # modelo, pero aca solo contamos; lo que pasa al modelo ya viene partido.
    return len(tokenizer(text, verbose=False)["input_ids"])


def split_for_model(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Parte el texto en fragmentos que el modelo pueda leer enteros.

    El modelo descarta todo lo que pase de max_tokens, y muchos resumenes son
    mas largos: sin esto perderian el final de la trama. Se corta entre
    oraciones, y entre palabras solo si una oracion sola no entra.
    """
    if count_tokens(tokenizer, text) <= max_tokens:
        return [text]

    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if count_tokens(tokenizer, sentence) <= max_tokens:
            pieces.append(sentence)
        else:
            pieces.extend(sentence.split())

    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and count_tokens(tokenizer, candidate) > max_tokens:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    chunks.append(current)
    return chunks


def build_embeddings(documents: list[str], ids: list[int]) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Cargando modelo {EMBEDDING_MODEL}...", file=sys.stderr)
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer, max_tokens = model.tokenizer, model.max_seq_length

    chunks_per_doc = [split_for_model(doc, tokenizer, max_tokens) for doc in documents]
    chunks = [chunk for doc_chunks in chunks_per_doc for chunk in doc_chunks]
    chunk_vectors = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Cada pelicula es el promedio de sus fragmentos, pesado por cantidad de
    # tokens: una oracion suelta al final no pesa lo mismo que un parrafo.
    matrix = np.empty((len(documents), chunk_vectors.shape[1]), dtype=np.float32)
    start = 0
    for row, doc_chunks in enumerate(chunks_per_doc):
        end = start + len(doc_chunks)
        weights = [count_tokens(tokenizer, chunk) for chunk in doc_chunks]
        mean = np.average(chunk_vectors[start:end], axis=0, weights=weights)
        # Normalizados a norma 1: el producto punto es la similitud coseno.
        matrix[row] = mean / np.linalg.norm(mean)
        start = end

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_INDEX, matrix=matrix, model=EMBEDDING_MODEL, ids=np.array(ids)
    )

    split = sum(len(doc_chunks) > 1 for doc_chunks in chunks_per_doc)
    print(
        f"Embeddings: {matrix.shape[0]} documentos x {matrix.shape[1]} dimensiones "
        f"({split} partidos en fragmentos de hasta {max_tokens} tokens)",
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
    ids = [movie["id"] for movie in movies]
    print(f"Corpus: {len(documents)} peliculas", file=sys.stderr)

    if args.method in ("tfidf", "both"):
        build_tfidf(documents, ids)
    if args.method in ("embeddings", "both"):
        build_embeddings(documents, ids)


if __name__ == "__main__":
    main()
