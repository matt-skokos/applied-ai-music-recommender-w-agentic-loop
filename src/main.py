"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("../data/songs.csv") 

    # Starter example profile
    user_prefs = {"genre": "indie", "mood": "chill", "energy": 0.3}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    header = f"Top {len(recommendations)} Recommendations"
    print(f"\n{header}\n{'=' * len(header)}\n")

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']}  —  score: {score:.2f}")
        for reason in explanation.split("; "):
            print(f"   - {reason}")
        print()


if __name__ == "__main__":
    main()
