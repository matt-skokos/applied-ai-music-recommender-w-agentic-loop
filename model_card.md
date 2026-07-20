# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

SongSeek V1.0

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

- This model will generate recommendations based on: genre, mood, tempo, valence, energy, danceability and acousticness
- This model makes assumptions that the user has a listening/liking history that has altered the defaults.
- This is for classroom exploration. 


---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What features does each `Song` use in your system
Content Features:
  1. Genre, coarsest, most reliable 
  2. Mood - direct categorical label
  3. Energy - continuous 0-1, easy to compare
  4. Tempo(BPM) - numerical and straightforward
  5. Valence - pairs with energy
  6. Danceability - continous 0-1, same shape as others
User Behavior features:
  7. Likes - clear signal
  8. Skips - another clear signal depending on listen-length
  9. Playlist adds - which songs "co-occur"
  10. Play count - another simple integer to imply signal strength
- What user preferences are considered 
The model uses vector-similarity so the same features are considered for a user's preferences.
- How does the model turn those into a score  
The model uses floating point integers from 0.00-1.00 and calculates a vector space scored for each dimension. 
- What changes did you make from the starter logic  
I added the vector similarity comparison and several dimensions to the scores

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
20 songs are representd in songs.csv catalog
- What genres or moods are represented  
r&b, hip-hop, r&b, hip-hop, classical, folk, house, indie pop, synthwave, lofi, ambiet, rock
- Did you add or remove data  
I added 6 new genres and 10 new songs to the catalog to better simulate operating conditions.
- Are there parts of musical taste missing in the dataset  
Yes, there are many genres and especially sub-genres that were left out to keep the sample simple enough to analyze.
---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results
Users who have a robust interaction history with the catalog will see better results.
- Any patterns you think your scoring captures correctly  
I think the most correct patterns are in song <-> user pref. space as these are the most easy to correlate for a user.
- Cases where the recommendations matched your intuition


---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
I tested 20 different profiles. 10 of them were generalized and represented "normal" users and 10 more represented adversarial and edge case users.
- What you looked for in the recommendations  
I primarily relied on the general score printout and especially weighted by judgement on the textual description of why a song was recommended. 
- What surprised you  
I was impressed by how many of the results were based solely on energy and mood ... I thought that several other variables would overtake these main components down the line but they remained the strongest classifiers. 
- Any simple tests or comparisons you ran  
I ran a suite of basic tests to confirm that recommendation was functional in small areas. I also developed some more generalized tests that spanned across many different user types and some that represented  edge case users.

No need for numeric metrics unless you created some.


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
