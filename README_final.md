# 🎵 Music Recommender Simulation

## Project Summary

This project is built off of the original Music Recommendation app that we designed and built for project 3. It includes an agentic loop that responds to user feedback. i.e. If a user "thumbs up" a song, this will skew the weights towards more similar songs and re-calculate the top-k results based on their feedback. The loop will also provide feedback with each move that it makes and will include more information from the Spotify API in their song ranking. 

## How The System Works

1. User queries song recommendations
2. Recommendations are returned, user responds to them with thumbs up or down
2. Agent recalculates user preferences, describes to user the changes
4. Recommendations are updated based on user input
5. User receives new recommendations
6. Loop starts at step 2

### Recommendation Logic
1. Input: UserProfile, Song List (ratings for each song)
2. Check against each song for vector similarity
3. Computer Similarity score -> Weighted Scores for each song returned + reasons
4. Take Top K songs 
5. Output ranked songs and reasons for recommend


---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---