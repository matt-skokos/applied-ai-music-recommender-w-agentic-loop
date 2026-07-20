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
1. **Profile #1** favors upbeat, high-energy pop, making it most similar to Profiles #6 and #7 in intensity but more cheerful and mainstream in style.

2. **Profile #2** emphasizes slow, chill, acoustic-leaning lofi, closely resembling Profile #4 while using more detailed tempo, valence, and danceability preferences.

3. **Profile #3** has the highest-intensity rock preference, sharing the energy of Profiles #1 and #6 but favoring a faster, heavier, and more aggressive sound.

4. **Profile #4** prefers the calmest and most acoustic ambient music, overlapping strongly with Profile #2 while being even lower in energy and less rhythm-focused.

5. **Profile #5** centers on relaxed, moderately danceable jazz, placing it between the quiet listening styles of Profiles #2 and #4 and the broader low-energy tastes of Profiles #8 and #9.

6. **Profile #6** prioritizes highly energetic and danceable house music, matching the intensity of Profiles #1 and #3 while being the most strongly oriented toward dancing.

7. **Profile #7** favors confident, high-energy, danceable hip-hop, resembling Profile #6 rhythmically but differing through its genre and assertive mood.

8. **Profile #8** prefers dreamy, highly acoustic classical music, sharing the low energy of Profiles #2 and #4 but emphasizing orchestral atmosphere rather than chill beats or ambient textures.

9. **Profile #9** combines folk and R&B at moderate-low energy, making it more genre-flexible than the other profiles while overlapping with Profiles #5 and #8 in its softer sound.

10. **Profile #10** represents the most balanced and genre-neutral profile, sitting near the middle of the group in energy, tempo, positivity, danceability, and acousticness.


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
I think that including a set of realistic songs and providing real-life users with a basic simulation to test would be helpful. The data that has been included and analyzed so far is just numerical and based on a bit of guesswork as to how it would perform in the real world.
- Better ways to explain recommendations  
I think that a bit more "flourish" in the textual explanations would help a lot. It's pretty bare-bones for testing purposes and basd on numerical stats, I think that some contextual info on the song/albums/genres would be helfpul.
- Improving diversity among the top results 
This is a tricky line to walk as it may cause results to become bifurcated ( some great, some terrible).
- Handling more complex user tastes  
I've tried to include a few data points that represent these users, again including real life songs, users with complex tastes would prove most fruitful in this endeavor.

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered
- How this changed the way you think about music recommendation apps  

I've noticed a striking similarity between music recommendation and web/data search. I was pleasantly surprised to be able to apply some knowledge I had about search engines and vector similarity search to the project. I also note that there is a deep level of subjective knowledge that is difficult to quantify and would benefit greatly from human input/guidance.

 What was your biggest learning moment during this project?
 I think that learning how to analyze and understand a complex process such as recommendation engines was my biggest learning moment. I feel like these crucial pieces of code are important to pick apart and understand before you just jump in and say "build this".
How did using AI tools help you, and when did you need to double-check them?
Using them helped me best when I was reviewing options for building the scoring algorithm. It was also a point where I needed to double check them as I had to rebuild some of what the original attempt created and make sure it made sense and didn't just pass tests.
What surprised you about how simple algorithms can still "feel" like recommendations?
I came up with what I thought wasn't exactly a simple solution: vector similarity scoring. But I guess in general it does simplify something like recommendations down to numbers and it was surprising how natural it could make the recommendations feel.
What would you try next if you extended this project?
I noted above that I'd try to test it with real world data and people and get some actual feedback on what I can improve upon.  I'd also test this on some really large datasets of songs. Furthermore I'd probably specialize this to work with genres that I personally enjoy and that I could come up with another component that can actually score songs on my own for features like tempo/vibe/genre.