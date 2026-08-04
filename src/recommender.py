import csv
import heapq
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


class Facet(str, Enum):
    """Tags a scoring reason with the dimension it came from, so callers can
    act on *which* facet mattered without re-parsing reason text."""
    GENRE = "genre"
    MOOD = "mood"
    ENERGY = "energy"
    TEMPO = "tempo"
    VALENCE = "valence"
    DANCEABILITY = "danceability"
    ACOUSTICNESS = "acousticness"


Reason = Tuple[str, Optional[Facet]]

# Scoring weights, in priority order: genre > mood > energy > tempo > valence > danceability > acousticness
W_GENRE = 0.30
W_MOOD = 0.22
W_ENERGY = 0.19
W_TEMPO = 0.14
W_VALENCE = 0.06
W_DANCEABILITY = 0.05
W_ACOUSTICNESS = 0.04

# tempo_bpm is on a raw BPM scale while every other target is 0.00-1.00,
# so it needs normalizing before it can be compared to target_tempo.
MIN_BPM = 60.0
MAX_BPM = 180.0

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Spotify-enriched genre tags (src/spotify_client.py), usually finer-grained
    # than `genre` above -- e.g. ["dream pop", "chillwave"] vs "lofi". Empty
    # when enrichment found no match or the artist has no Spotify genres.
    spotify_genres: List[str] = field(default_factory=list)

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    user_id: str
    # Categorical favorites, used for genre/mood match bonuses
    favorite_genres: Set[str] = field(default_factory=set)
    favorite_moods: Set[str] = field(default_factory=set)
    # Numeric taste vector, all values 0.00-1.00, used for content similarity
    target_energy: float = 0.5
    target_valence: float = 0.5
    target_danceability: float = 0.5
    target_acousticness: float = 0.5
    target_tempo: float = 0.5
    # Interaction history, keyed by song id
    likes: Set[int] = field(default_factory=set)
    skips: Dict[int, int] = field(default_factory=dict)
    playlist_adds: Set[int] = field(default_factory=set)
    play_counts: Dict[int, int] = field(default_factory=dict)

def _normalize_tempo(tempo_bpm: float) -> float:
    """Scales a raw BPM value to the 0.00-1.00 range used by target_tempo."""
    return max(0.0, min(1.0, (tempo_bpm - MIN_BPM) / (MAX_BPM - MIN_BPM)))

def _score(
    genre_match: bool,
    mood_match: bool,
    energy: float,
    tempo_bpm: float,
    valence: float,
    danceability: float,
    acousticness: float,
    target_energy: float,
    target_tempo: float,
    target_valence: float,
    target_danceability: float,
    target_acousticness: float,
    genre_label: str,
    mood_label: str,
) -> Tuple[float, List[Reason]]:
    """
    Shared scoring core used by both the OOP (Song/UserProfile) and
    functional (dict-based) recommendation paths, so the weights and
    formula only live in one place.

    Each reason is tagged with the Facet it came from (or None for the
    catch-all), so callers that need to act on *which* facet mattered
    (e.g. nudging session weights from feedback) don't have to re-parse
    reason text against a separately maintained string->Facet mapping.
    """
    tempo_sim = 1 - abs(target_tempo - _normalize_tempo(tempo_bpm))
    energy_sim = 1 - abs(target_energy - energy)
    valence_sim = 1 - abs(target_valence - valence)
    dance_sim = 1 - abs(target_danceability - danceability)
    acoustic_sim = 1 - abs(target_acousticness - acousticness)

    score = (
        W_GENRE * (1.0 if genre_match else 0.0)
        + W_MOOD * (1.0 if mood_match else 0.0)
        + W_ENERGY * energy_sim
        + W_TEMPO * tempo_sim
        + W_VALENCE * valence_sim
        + W_DANCEABILITY * dance_sim
        + W_ACOUSTICNESS * acoustic_sim
    )

    reasons: List[Reason] = []
    if genre_match:
        reasons.append((f"matches your favorite genre ({genre_label})", Facet.GENRE))
    if mood_match:
        reasons.append((f"matches your favorite mood ({mood_label})", Facet.MOOD))
    if energy_sim >= 0.85:
        reasons.append(("energy closely matches your taste", Facet.ENERGY))
    if tempo_sim >= 0.85:
        reasons.append(("tempo closely matches your taste", Facet.TEMPO))
    if valence_sim >= 0.85:
        reasons.append(("valence closely matches your taste", Facet.VALENCE))
    if dance_sim >= 0.85:
        reasons.append(("danceability closely matches your taste", Facet.DANCEABILITY))
    if acoustic_sim >= 0.85:
        reasons.append(("acousticness closely matches your taste", Facet.ACOUSTICNESS))
    if not reasons:
        reasons.append(("a reasonable overall match to your taste profile", None))

    return score, reasons

def _score_song_for_user(user: UserProfile, song: Song) -> Tuple[float, List[Reason]]:
    """Adapts a Song/UserProfile pair into the shared _score() scoring core."""
    genre_match = song.genre in user.favorite_genres or any(
        g in user.favorite_genres for g in song.spotify_genres
    )
    return _score(
        genre_match=genre_match,
        mood_match=song.mood in user.favorite_moods,
        energy=song.energy,
        tempo_bpm=song.tempo_bpm,
        valence=song.valence,
        danceability=song.danceability,
        acousticness=song.acousticness,
        target_energy=user.target_energy,
        target_tempo=user.target_tempo,
        target_valence=user.target_valence,
        target_danceability=user.target_danceability,
        target_acousticness=user.target_acousticness,
        genre_label=song.genre,
        mood_label=song.mood,
    )

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Stores the candidate songs to recommend from."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top k songs for the user, sorted from highest to lowest score."""
        return heapq.nlargest(k, self.songs, key=lambda song: _score_song_for_user(user, song)[0])

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable string explaining why the song was recommended."""
        _, reasons = _score_song_for_user(user, song)
        return "; ".join(text for text, _ in reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file. Returns a list of Dicts of song values.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                "likes": int(row["likes"]),
                "skips": int(row["skips"]),
                "playlist_adds": int(row["playlist_adds"]),
                "play_count": int(row["play_count"]),
            })
    print(f'Loaded {len(songs)} number of songs successfully.')
    return songs

_SONG_FIELDS = {"id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence", "danceability", "acousticness"}

def load_songs_as_objects(csv_path: str) -> List[Song]:
    """Loads songs.csv into Song objects for the OOP path (Recommender, rebuild_pool, interpret_feedback); spotify_genres defaults to empty until merged with a genre cache."""
    return [Song(**{k: v for k, v in row.items() if k in _SONG_FIELDS}) for row in load_songs(csv_path)]

def _as_favorite_set(value) -> Set[str]:
    """Normalizes a genre/mood preference (None, str, or collection) into a
    set."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return set(value)

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[Reason]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    favorite_genres = _as_favorite_set(user_prefs.get("genre", user_prefs.get("favorite_genres")))
    favorite_moods = _as_favorite_set(user_prefs.get("mood", user_prefs.get("favorite_moods")))

    return _score(
        genre_match=song["genre"] in favorite_genres,
        mood_match=song["mood"] in favorite_moods,
        energy=float(song["energy"]),
        tempo_bpm=float(song["tempo_bpm"]),
        valence=float(song["valence"]),
        danceability=float(song["danceability"]),
        acousticness=float(song["acousticness"]),
        target_energy=float(user_prefs.get("energy", 0.5)),
        target_tempo=float(user_prefs.get("tempo", 0.5)),
        target_valence=float(user_prefs.get("valence", 0.5)),
        target_danceability=float(user_prefs.get("danceability", 0.5)),
        target_acousticness=float(user_prefs.get("acousticness", 0.5)),
        genre_label=song["genre"],
        mood_label=song["mood"],
    )

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(text for text, _ in reasons)))
    return heapq.nlargest(k, scored, key=lambda item: item[1])
