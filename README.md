# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  I'll separate into Content Features:
  1. Genre, coarsest, most reliable 
  2. Mood - direct categorical label
  3. Energy - continuous 0-1, easy to compare
  4. Tempo(BPM) - numerical and straightforward
  5. Valence - pairs with energy
  6. Danceability - continous 0-1, same shape as others

  and User Behavior features:
  7. Likes - clear signal
  8. Skips - another clear signal depending on listen-length
  9. Playlist adds - which songs "co-occur"
  10. Play count - another simple integer to imply signal strength

  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
I'll need to store 2 different types of user data in the UserProfile.
The first is a "taste" vector. It describes what type of content the user will rank closely to with content that has similar genre, energy, mood, etc.
The second is an interaction history. This will map different user actions. It will help to weight song profiles that have been repeated, not skipped and such.
- How does your `Recommender` compute a score for each song
The recommender will heavily lean on vector-similarity scoring. This is to say that the euclidean distance or angle between the two  in a vector space, with very similar profile vectors will create a match between them. The closer this song, the sooner it will be recommended to the user.  
- How do you choose which songs to recommend
At a very basic level the recommendations are based on vector-similarity as noted above. I will also include a flag for songs that haven't been heard before as being recommended first.

You can include a simple diagram or bullet list if helpful.

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

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



