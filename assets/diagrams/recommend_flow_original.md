flowchart TD
    A[Inputs: songs list + user prefs/UserProfile] --> B{For each song}
    B --> C[Check genre match & mood match]
    C --> D[Compute similarity: energy, tempo, valence, danceability, acousticness]
    D --> E[Weighted score_song → score, reasons]
    E --> B
    B -->|all songs scored| F[Sort by score, descending]
    F --> G[Take top k]
    G --> H[Output: ranked songs + explanations]
