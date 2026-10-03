const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const kSelect = document.getElementById("k");
const genreSelect = document.getElementById("genre");
const decadeSelect = document.getElementById("decade");
const statsEl = document.getElementById("stats");
const statusEl = document.getElementById("status");
const announcer = document.getElementById("announcer");
const columnsEl = document.getElementById("columns");
const cardTemplate = document.getElementById("card-template");

const columns = {
  tfidf: document.querySelector('[data-method="tfidf"]'),
  embeddings: document.querySelector('[data-method="embeddings"]'),
};

// Con menos letras, buscar mientras se escribe da resultados sin sentido.
const MIN_TYPED_LENGTH = 3;

// Solo cuenta la ultima busqueda: si llega una nueva, se cancela la anterior
// para que una respuesta vieja no pise a la nueva.
let activeRequest = null;
let corpusSize = null;
// true si la entrada actual del historial la creo la escritura en curso: las
// teclas siguientes la actualizan en vez de apilar una entrada por tecla.
let typingEntry = false;

function debounce(fn, ms) {
  let timer;
  const debounced = () => {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
  debounced.cancel = () => clearTimeout(timer);
  return debounced;
}

// --- Estado: la URL es la fuente de verdad, asi Atras/Adelante funcionan ---

function readForm() {
  return {
    q: queryInput.value.trim(),
    k: kSelect.value,
    genre: genreSelect.value,
    decade: decadeSelect.value,
  };
}

function toParams(state) {
  const params = new URLSearchParams({ q: state.q, k: state.k });
  if (state.genre) params.set("genre", state.genre);
  if (state.decade) params.set("decade", state.decade);
  return params;
}

function applyUrlToForm() {
  const params = new URLSearchParams(location.search);
  queryInput.value = params.get("q") ?? "";
  setSelect(kSelect, params.get("k"));
  setSelect(genreSelect, params.get("genre"));
  setSelect(decadeSelect, params.get("decade"));
}

// Si el valor no es una opcion valida (una URL editada a mano), queda la primera.
function setSelect(select, value) {
  const exists = [...select.options].some((option) => option.value === value);
  select.value = exists ? value : select.options[0].value;
}

function updateHistory(params, mode) {
  const url = `?${params}`;
  if (mode === "push" && url !== location.search) {
    history.pushState(null, "", url);
  } else if (mode !== "none") {
    history.replaceState(null, "", url);
  }
}

// --- Datos del servidor ---------------------------------------------------

async function loadInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) return;
    const info = await response.json();
    corpusSize = info.movies;
    columns.tfidf.querySelector("[data-dims]").textContent = info.tfidf_dims;
    columns.embeddings.querySelector("[data-dims]").textContent = info.embedding_dims;
    for (const { name, count } of info.genres) {
      genreSelect.append(new Option(`${name} (${count})`, name));
    }
    for (const { decade, count } of info.decades) {
      decadeSelect.append(new Option(`${decadeLabel(decade)} (${count})`, decade));
    }
    showStats(corpusSize);
  } catch {
    // Sin esta informacion la busqueda funciona igual, solo que sin filtros.
  }
}

function decadeLabel(decade) {
  return decade >= 2000 ? `Años ${decade}` : `Años ${String(decade).slice(2)}`;
}

async function search(state, historyMode) {
  activeRequest?.abort();
  const request = new AbortController();
  activeRequest = request;

  const params = toParams(state);
  updateHistory(params, historyMode);
  statusEl.textContent = "";
  columnsEl.setAttribute("aria-busy", "true");
  // Una busqueda suele tardar milisegundos: las columnas se atenuan solo si
  // demora, para que no parpadeen mientras se escribe.
  const dimTimer = setTimeout(() => columnsEl.classList.add("loading"), 150);

  try {
    const response = await fetch(`/api/search?${params}`, { signal: request.signal });
    if (!response.ok) throw new Error(`el servidor respondió ${response.status}`);
    const data = await response.json();
    const empty = emptyMessage(data.candidates, Boolean(state.genre || state.decade));
    for (const [key, column] of Object.entries(columns)) {
      renderColumn(column, key, data.methods[key], empty);
    }
    updateMoreButtons();
    showStats(data.candidates);
    announce(data, state.q);
  } catch (error) {
    if (error.name === "AbortError") return;
    statusEl.textContent = `No se pudo buscar: ${error.message}. ¿Está corriendo el servidor?`;
  } finally {
    clearTimeout(dimTimer);
    if (activeRequest === request) {
      columnsEl.classList.remove("loading");
      columnsEl.setAttribute("aria-busy", "false");
    }
  }
}

// --- Render ---------------------------------------------------------------

function emptyMessage(candidates, filtered) {
  if (candidates === 0) return "Ninguna película cumple los filtros.";
  return filtered
    ? "Ninguna película del filtro comparte términos con la consulta."
    : "Ninguna película comparte términos con la consulta.";
}

function renderColumn(column, key, method, emptyText) {
  column.querySelector("[data-ms]").textContent = `${method.ms} ms`;
  const list = column.querySelector(".results");
  if (method.results.length === 0) {
    showEmpty(list, emptyText);
    return;
  }
  const best = method.results[0].score;
  list.replaceChildren(
    ...method.results.map((movie, i) => renderCard(movie, `${key}-${i + 1}`, i + 1, best)),
  );
}

function showEmpty(list, text) {
  const item = document.createElement("li");
  item.className = "empty";
  item.textContent = text;
  list.replaceChildren(item);
}

function renderCard(movie, id, rank, best) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);

  card.querySelector(".rank").textContent = rank;
  card.querySelector(".title").textContent = movie.title;
  card.querySelector(".score").textContent = movie.score.toFixed(4);

  const meta = [movie.year ?? "s/f"];
  if (movie.vote_average) meta.push(`★ ${movie.vote_average.toFixed(1)}`);
  card.querySelector(".meta").textContent = meta.join(" · ");

  const genres = card.querySelector(".genres");
  for (const genre of movie.genres) {
    const item = document.createElement("li");
    item.textContent = genre;
    genres.append(item);
  }

  const overview = card.querySelector(".overview");
  overview.id = `overview-${id}`;
  overview.textContent = movie.overview;
  card.querySelector(".more").setAttribute("aria-controls", overview.id);

  // Relativa al primero de la columna: muestra cuanto se separa cada resultado
  // del mejor, que es lo comparable dentro de un mismo metodo.
  const ratio = best > 0 ? Math.max(0, movie.score / best) : 0;
  card.querySelector(".bar span").style.width = `${ratio * 100}%`;

  setPoster(card.querySelector(".poster"), movie);
  return card;
}

function setPoster(container, movie) {
  const showInitial = () => {
    container.classList.add("no-poster");
    container.textContent = movie.title.charAt(0);
  };
  if (!movie.poster_url) {
    showInitial();
    return;
  }
  const img = container.querySelector("img");
  img.addEventListener("error", showInitial, { once: true });
  img.src = movie.poster_url;
}

// "Ver más" aparece solo si el resumen no entra en las lineas visibles. Se
// mide todo primero y se escribe despues, para no forzar un layout por tarjeta.
function updateMoreButtons() {
  const cards = [...columnsEl.querySelectorAll(".card:not(.expanded)")];
  const clamped = cards.map((card) => {
    const overview = card.querySelector(".overview");
    return overview.scrollHeight > overview.clientHeight + 1;
  });
  cards.forEach((card, i) => {
    card.querySelector(".more").hidden = !clamped[i];
  });
}

function showStats(candidates) {
  if (corpusSize === null) return;
  const count = document.createElement("strong");
  if (candidates === corpusSize) {
    count.textContent = corpusSize;
    statsEl.replaceChildren("Corpus: ", count, " películas de TMDB");
  } else {
    count.textContent = candidates;
    statsEl.replaceChildren("Con los filtros: ", count, ` de ${corpusSize} películas`);
  }
  statsEl.hidden = false;
}

function announce(data, query) {
  const tfidf = data.methods.tfidf.results.length;
  const embeddings = data.methods.embeddings.results.length;
  announcer.textContent =
    `${tfidf} resultados de TF-IDF y ${embeddings} de embeddings para «${query}».`;
}

function resetResults() {
  activeRequest?.abort();
  statusEl.textContent = "";
  // Si quedara el anuncio viejo, volver a esa misma busqueda no se anunciaria.
  announcer.textContent = "";
  for (const column of Object.values(columns)) {
    column.querySelector("[data-ms]").textContent = "";
    showEmpty(column.querySelector(".results"), "Los resultados aparecen acá.");
  }
  showStats(corpusSize);
}

// --- Eventos --------------------------------------------------------------

const searchWhileTyping = debounce(() => {
  const state = readForm();
  // Si la URL ya es esta busqueda (se escribio y se borro), no hay nada nuevo.
  if (state.q.length < MIN_TYPED_LENGTH || `?${toParams(state)}` === location.search) return;
  search(state, typingEntry ? "replace" : "push");
  typingEntry = true;
}, 300);

// Enter, un ejemplo o un filtro: busqueda nueva, con su propia entrada.
function commitSearch() {
  searchWhileTyping.cancel();
  typingEntry = false;
  const state = readForm();
  if (state.q) search(state, "push");
}

queryInput.addEventListener("input", searchWhileTyping);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  commitSearch();
});

for (const select of [genreSelect, decadeSelect, kSelect]) {
  select.addEventListener("change", commitSearch);
}

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.example;
    commitSearch();
  });
}

columnsEl.addEventListener("click", (event) => {
  const button = event.target.closest(".more");
  if (!button) return;
  const expanded = button.closest(".card").classList.toggle("expanded");
  button.setAttribute("aria-expanded", String(expanded));
  button.textContent = expanded ? "Ver menos" : "Ver más";
});

// Al cambiar el ancho cambia que resumenes entran enteros.
window.addEventListener("resize", debounce(updateMoreButtons, 150));

// Atras/Adelante: la URL ya cambio, solo hay que reflejarla.
window.addEventListener("popstate", () => {
  searchWhileTyping.cancel();
  typingEntry = false;
  applyUrlToForm();
  const state = readForm();
  if (state.q) search(state, "none");
  else resetResults();
});

async function init() {
  // Primero la info: los filtros de la URL necesitan las opciones cargadas.
  await loadInfo();
  applyUrlToForm();
  const state = readForm();
  if (state.q) search(state, "replace");
}

init();
