"""
Prints main.py-style Top-5 recommendations for the 10 "diverse" user
profiles in data/user_profiles.csv (category == "diverse"), so they can be
compared side by side without editing main.py's hardcoded profile.
"""

import csv

from recommender import load_songs, recommend_songs

PROFILES_CSV_PATH = "../data/user_profiles.csv"
SONGS_CSV_PATH = "../data/songs.csv"


def _row_to_prefs(row: dict) -> dict:
    """Converts one data/user_profiles.csv row into a recommend_songs() prefs dict."""
    prefs = {}

    genre = row["genre"]
    if genre:
        if "|" in genre:
            prefs["favorite_genres"] = genre.split("|")
        else:
            prefs["genre"] = genre

    mood = row["mood"]
    if mood:
        prefs["mood"] = mood

    for field in ("energy", "tempo", "valence", "danceability", "acousticness"):
        value = row[field]
        if value:
            prefs[field] = float(value)

    return prefs


def main() -> None:
    """Prints top-5 recommendations for each "diverse" profile in user_profiles.csv, side by side."""
    songs = load_songs(SONGS_CSV_PATH)

    with open(PROFILES_CSV_PATH, newline="") as f:
        profile_rows = [row for row in csv.DictReader(f) if row["category"] == "diverse"]

    for row in profile_rows:
        prefs = _row_to_prefs(row)
        recommendations = recommend_songs(prefs, songs, k=5)

        header = f"Profile #{row['profile_id']}: {prefs}"
        print(f"\n{header}\n{'=' * len(header)}\n")

        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"{rank}. {song['title']}  —  score: {score:.2f}")
            for reason in explanation.split("; "):
                print(f"   - {reason}")
            print()


if __name__ == "__main__":
    main()
