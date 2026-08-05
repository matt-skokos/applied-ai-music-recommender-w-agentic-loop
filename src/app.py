"""
Streamlit interface for the agentic recommendation loop: browse the
catalog, generate a playlist, thumbs up/down each song, watch the
transparency message and the next round's playlist update. This is the
first real end-to-end exercise of session.py/llm.py's building blocks.
Run with: streamlit run src/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# streamlit run's sys.path handling isn't consistent across how the script
# gets invoked (real CLI run vs. AppTest harness), so pin it explicitly
# rather than relying on whichever behavior happens to be incidental.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recommender import Song, UserProfile, load_songs_as_objects
from session import FeedbackType, RecommendationSession, interpret_feedback, rebuild_pool
from llm import build_transparency_message
from spotify_client import apply_genre_cache

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SONGS_CSV_PATH = _DATA_DIR / "songs.csv"
_GENRE_CACHE_PATH = _DATA_DIR / "spotify_genres_cache.json"
_TOP_K = 5
_SHOW_MORE_STEP = 5
_MAX_SHOW_MORE_CLICKS = 2
_MAX_K = _TOP_K + _MAX_SHOW_MORE_CLICKS * _SHOW_MORE_STEP


@st.cache_data
def load_catalog() -> list[Song]:
    """Loads the catalog once per app run and merges in any cached Spotify genres."""
    songs = load_songs_as_objects(str(_SONGS_CSV_PATH))
    return apply_genre_cache(songs, str(_GENRE_CACHE_PATH))


def catalog_dataframe(songs: list[Song]) -> pd.DataFrame:
    """Flattens the catalog into a DataFrame for the browsable table."""
    return pd.DataFrame([
        {
            "title": s.title, "artist": s.artist, "genre": s.genre, "mood": s.mood,
            "energy": s.energy, "tempo_bpm": s.tempo_bpm, "valence": s.valence,
            "danceability": s.danceability, "acousticness": s.acousticness,
        }
        for s in songs
    ])


def _reset_playlist_view() -> None:
    """Collapses the visible list back to just the top-k after a new playlist/round -- "show more" starts over each round."""
    st.session_state.visible_count = _TOP_K
    st.session_state.show_more_clicks = 0


def render_profile_form(catalog: list[Song]) -> None:
    """Renders the taste-profile form; saves a UserProfile to session_state on submit."""
    genres = sorted({s.genre for s in catalog})
    moods = sorted({s.mood for s in catalog})

    with st.form("profile_form"):
        st.subheader("Your taste profile")
        favorite_genres = st.multiselect("Favorite genres", genres)
        favorite_moods = st.multiselect("Favorite moods", moods)
        col1, col2, col3 = st.columns(3)
        target_energy = col1.slider("Energy", 0.0, 1.0, 0.5, 0.05)
        target_valence = col2.slider("Valence (positivity)", 0.0, 1.0, 0.5, 0.05)
        target_danceability = col3.slider("Danceability", 0.0, 1.0, 0.5, 0.05)
        col4, col5 = st.columns(2)
        target_acousticness = col4.slider("Acousticness", 0.0, 1.0, 0.5, 0.05)
        target_tempo = col5.slider("Tempo (0=slow, 1=fast)", 0.0, 1.0, 0.5, 0.05)

        if st.form_submit_button("Save profile"):
            st.session_state.user_profile = UserProfile(
                user_id="streamlit_user",
                favorite_genres=set(favorite_genres),
                favorite_moods=set(favorite_moods),
                target_energy=target_energy,
                target_valence=target_valence,
                target_danceability=target_danceability,
                target_acousticness=target_acousticness,
                target_tempo=target_tempo,
            )
            st.session_state.session = None
            st.session_state.current_topk = None
            st.session_state.transparency_message = None
            _reset_playlist_view()


def handle_feedback(song: Song, reasons, feedback: FeedbackType, catalog: list[Song]) -> None:
    """Applies one thumbs up/down: nudges weights, gets the transparency message, rebuilds the next round's top-k."""
    session = st.session_state.session
    interpret_feedback(session, song, reasons, feedback)
    session.round_number += 1
    message, _log = build_transparency_message(session)
    st.session_state.transparency_message = message
    st.session_state.current_topk, _log = rebuild_pool(session, catalog, k=_MAX_K)
    _reset_playlist_view()


def render_playlist_section(catalog: list[Song]) -> None:
    """Renders the Generate Playlist button, the current top-k with thumbs up/down, and the transparency message."""
    st.subheader("Your playlist")

    if st.session_state.user_profile is None:
        st.info("Save a taste profile above first.")
        return

    if st.button("Generate playlist"):
        if st.session_state.session is None:
            st.session_state.session = RecommendationSession(base_user=st.session_state.user_profile)
        st.session_state.current_topk, _log = rebuild_pool(st.session_state.session, catalog, k=_MAX_K)
        st.session_state.transparency_message = None
        _reset_playlist_view()

    if st.session_state.transparency_message:
        st.info(st.session_state.transparency_message)

    topk = st.session_state.current_topk
    if topk is None:
        return
    if not topk:
        st.warning("No songs left to recommend -- you've thumbs-downed the whole catalog.")
        return

    round_number = st.session_state.session.round_number
    visible = topk[: st.session_state.visible_count]

    st.caption(f"Round {round_number} -- showing {len(visible)} of {len(topk)}")
    for song, score, reasons in visible:
        cols = st.columns([5, 1, 1])
        explanation = "; ".join(text for text, _facet in reasons)
        cols[0].markdown(f"**{song.title}** -- {song.artist}  \n"
                          f"_{song.genre}, {song.mood}_ -- score {score:.2f}  \n"
                          f"{explanation}")
        if cols[1].button("\U0001F44D", key=f"up_{song.id}_{round_number}"):
            handle_feedback(song, reasons, FeedbackType.UP, catalog)
            st.rerun()
        if cols[2].button("\U0001F44E", key=f"down_{song.id}_{round_number}"):
            handle_feedback(song, reasons, FeedbackType.DOWN, catalog)
            st.rerun()

    can_show_more = (
        st.session_state.show_more_clicks < _MAX_SHOW_MORE_CLICKS
        and st.session_state.visible_count < len(topk)
    )
    if can_show_more:
        if st.button(f"Show {_SHOW_MORE_STEP} more"):
            st.session_state.visible_count = min(st.session_state.visible_count + _SHOW_MORE_STEP, len(topk))
            st.session_state.show_more_clicks += 1
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="SongSeek", layout="wide")
    st.title("SongSeek")

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
        st.session_state.session = None
        st.session_state.current_topk = None
        st.session_state.transparency_message = None
        _reset_playlist_view()

    catalog = load_catalog()

    render_profile_form(catalog)

    with st.expander(f"Browse the catalog ({len(catalog)} songs)"):
        st.table(catalog_dataframe(catalog))

    if st.button("Start over"):
        st.session_state.user_profile = None
        st.session_state.session = None
        st.session_state.current_topk = None
        st.session_state.transparency_message = None
        _reset_playlist_view()
        st.rerun()

    render_playlist_section(catalog)


if __name__ == "__main__":
    main()
