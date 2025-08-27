
import os
import re
from pathlib import Path
import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

# ----------------------
# Konfiguracja plików
# ----------------------
MODEL_NAME = "welcome_survey_clustering_pipeline_v2"
DATA = "welcome_survey_simple_v2.csv"

# >>> Zmienione: obsługa właściwej nazwy pliku JSON (z/bez .json)
JSON_CANDIDATES = [
    "welcome_survey_cluster_names_and_descriptions_with_images.json",
    "welcome_survey_cluster_names_and_descriptions_with_images",
    # fallbacki na wypadek starej nazwy:
    "welcome_survey_cluster_names_and_descriptions_v2.json",
    "welcome_survey_cluster_names_and_descriptions_v2",
]

# ----------------------
# USTAWIENIA ŚCIEŻEK OBRAZÓW
# ----------------------
ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

BASE_DIR = Path(__file__).parent.resolve()
CWD_DIR = Path.cwd().resolve()

# Pozwól nadpisać lokalizację przez zmienną środowiskową
ENV_DIR = os.environ.get("CLUSTER_IMAGE_DIR", "").strip()

candidate_dirs = [
    Path(ENV_DIR) if ENV_DIR else None,
    BASE_DIR / "cluster_images",
    BASE_DIR / "images",
    CWD_DIR / "cluster_images",
    CWD_DIR / "images",
    BASE_DIR.parent / "cluster_images",
    BASE_DIR.parent / "images",
]

# Odfiltruj None, istniejące katalogi i usuń duplikaty zachowując kolejność
seen = set()
CANDIDATE_DIRS = []
for p in candidate_dirs:
    if p and p.is_dir():
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            CANDIDATE_DIRS.append(p)

# Jeśli nic nie znaleziono, ustaw domyślnie BASE_DIR/cluster_images (może zostać utworzone gdzieś indziej)
if not CANDIDATE_DIRS:
    CANDIDATE_DIRS = [BASE_DIR / "cluster_images"]


def _index_images(dirs):
    index = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                stem = p.stem
                if stem not in index:
                    index[stem] = str(p.resolve())
    return index


IMAGE_INDEX = _index_images(CANDIDATE_DIRS)


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


def _find_json_path():
    """Zwróć pierwszą istniejącą ścieżkę JSON z listy kandydatów (obok app.py lub w CWD)."""
    # Najpierw obok app.py
    for name in JSON_CANDIDATES:
        p = BASE_DIR / name
        if p.is_file():
            return str(p)
    # Potem w katalogu uruchomienia
    for name in JSON_CANDIDATES:
        p = CWD_DIR / name
        if p.is_file():
            return str(p)
    return None


@st.cache_data
def get_cluster_names_and_descriptions():
    json_path = _find_json_path()
    if not json_path:
        st.error(
            "Nie znaleziono pliku JSON z opisami klastrów. "
            "Sprawdź, czy istnieje plik: "
            f"{', '.join(JSON_CANDIDATES)} (w katalogu aplikacji lub bieżącym)."
        )
        st.stop()
    with open(json_path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants(_model_key: str = MODEL_NAME):
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


def _normalize_cluster_id(cluster_id) -> str:
    s = str(cluster_id).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(int(round(f)))
    except Exception:
        pass
    m = re.search(r'\d+', s)
    if m:
        return m.group(0)
    return s


def _get_cluster_record(cluster_cfg, cluster_id_norm: str):
    if isinstance(cluster_cfg, dict):
        if cluster_id_norm in cluster_cfg:
            return cluster_cfg[cluster_id_norm]
        key_with_prefix = f"Cluster {cluster_id_norm}"
        if key_with_prefix in cluster_cfg:
            return cluster_cfg[key_with_prefix]
        try:
            cid_int = int(cluster_id_norm)
            if cid_int in cluster_cfg:
                return cluster_cfg[cid_int]
        except Exception:
            pass
        for k in cluster_cfg.keys():
            ks = str(k)
            m = re.search(r'\d+', ks)
            if m and m.group(0) == cluster_id_norm:
                return cluster_cfg[k]
        return None
    elif isinstance(cluster_cfg, list):
        try:
            idx = int(cluster_id_norm)
            if 0 <= idx < len(cluster_cfg):
                return cluster_cfg[idx]
        except Exception:
            pass
    return None


def resolve_image_path(cluster_data: dict, cluster_id):
    cid = _normalize_cluster_id(cluster_id)
    img_field = (cluster_data.get("image", "") or "").strip()

    if img_field:
        base = os.path.basename(img_field)
        stem, ext = os.path.splitext(base)
        for d in CANDIDATE_DIRS:
            p = d / base
            if p.is_file():
                return str(p.resolve())
        if stem in IMAGE_INDEX:
            return IMAGE_INDEX[stem]

    for s in (f"cluster_{cid}", f"cluster_{cid}_2"):
        if s in IMAGE_INDEX:
            return IMAGE_INDEX[s]

    return None


def render_cluster_image(predicted_cluster_data: dict, predicted_cluster_id, debug: bool = False):
    path = resolve_image_path(predicted_cluster_data, predicted_cluster_id)
    if path:
        st.image(path, caption=predicted_cluster_data.get("name", "Klaster"), use_container_width=True)
        if debug:
            st.caption(f"Użyty obraz: {path}")
    else:
        if debug:
            with st.expander("Debug obrazków (kliknij)"):
                st.write("Katalogi przeszukiwane:", [str(p) for p in CANDIDATE_DIRS])
                st.write("Indeks obrazów (stem → ścieżka):", list(IMAGE_INDEX.items()))
                st.write("Wartość 'image' w JSON:", predicted_cluster_data.get("image", None))
                st.write("Znormalizowany cluster_id:", _normalize_cluster_id(predicted_cluster_id))
        st.info("Obrazek dla tego klastra nie został odnaleziony.")


# ----------------------
# Aplikacja
# ----------------------
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ["<18", "25-34", "45-54", "35-44", "18-24", ">=65", "55-64", "unknown"])
    edu_level = st.selectbox("Wykształcenie", ["Podstawowe", "Średnie", "Wyższe"])
    fav_animals = st.selectbox("Ulubione zwierzęta", ["Brak ulubionych", "Psy", "Koty", "Inne", "Koty i Psy"])
    fav_place = st.selectbox("Ulubione miejsce", ["Nad wodą", "W lesie", "W górach", "Inne"])
    gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])

    person_df = pd.DataFrame(
        [
            {
                "age": age,
                "edu_level": edu_level,
                "fav_animals": fav_animals,
                "fav_place": fav_place,
                "gender": gender,
            }
        ]
    )

model = get_model()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
all_df = get_all_participants(MODEL_NAME)

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
cid_norm = _normalize_cluster_id(predicted_cluster_id)
predicted_cluster_data = _get_cluster_record(cluster_names_and_descriptions, cid_norm)

if not predicted_cluster_data:
    st.error(f"Nie udało się pobrać opisu klastra dla id='{predicted_cluster_id}' (znormalizowane: '{cid_norm}'). "
             f"Dostępne klucze: {list(cluster_names_and_descriptions.keys())}")
    st.stop()

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
render_cluster_image(predicted_cluster_data, cid_norm, debug=False)
st.markdown(predicted_cluster_data["description"])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")

fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(title="Rozkład wieku w grupie", xaxis_title="Wiek", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(title="Rozkład wykształcenia w grupie", xaxis_title="Wykształcenie", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(title="Rozkład ulubionych zwierząt w grupie", xaxis_title="Ulubione zwierzęta", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(title="Rozkład ulubionych miejsc w grupie", xaxis_title="Ulubione miejsce", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(title="Rozkład płci w grupie", xaxis_title="Płeć", yaxis_title="Liczba osób")
st.plotly_chart(fig)
