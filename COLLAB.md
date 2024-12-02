# **Branching Instructions for Group Members**

To ensure smooth collaboration, each group member will work on their own branch. Here’s how to set it up:

---

## **Branch Names**
For example:
- **Melis**: Use the branch name `melis`

---

## **Steps to Create & Use Your Branch**


### **1. Switch to Your Branch**
Run the following command to create and switch to your branch:
```bash
git checkout -b <branch-name>
```

### **2. Make Your Changes**
After making your changes, stage and commit them:
```bash
git add .
git commit -m "Describe your changes here"
```
### **3. Push Your Changes**
Push your changes to the remote repository:
```bash
git push origin <branch-name>
```

### **4. Sync With the Main Branch**
Before merging your work, make sure your branch is up to date with the latest changes from main:
```bash
git checkout main
git pull origin main
git checkout <branch-name>
git merge main
```

_______
# Next Steps to Complete the Project

Based on the notebooks we've created and the current state of the latex file, the following should be conducted to finish the project

---

## 1. Dataset Finalization and Cleaning (``Johan``)
- **Complete missing actor-character mappings**: Use the methods in `actor_webscraping.pdf` to further refine unmatched characters and validate the data between Cornell and TMDB datasets.
- **Clean and structure genres**: Ensure consistent formatting and remove duplicates or ambiguous entries in the genre columns.

---

## 2. Network Construction (``Johan``)
- **Bipartite Graph Creation**:
  - Finalize the actor-movie bipartite graph construction.
  - Use projections for actor-actor and movie-movie networks, with edges weighted by shared collaborations.

---

## 3. Network Metrics and Centrality Analysis (``Johan``)
- **Compute network statistics**:
  - Actor-actor projection: Analyze degree, betweenness, and eigenvector centralities to identify influential actors.
  - Movie-movie projection: Use thresholds (e.g., shared actors >1, >2) to explore structural changes.

---

## 4. Community Detection (``Johan``)
- **Apply the Louvain algorithm** to identify communities in the networks:
  - Actor-actor network: Look for genre-based clusters (e.g., actors working in the same genre or franchise).
  - Movie-movie network: Find clusters indicative of production or thematic links.

---

## 5. Temporal and Sentiment Analysis (``Melis``)
- **Sentiment Over Time**:
  - Refine the temporal sentiment analysis to include cumulative sentiment arcs for key movies or genres.
  - Use decade-based trends to link sentiment changes to cultural or historical contexts.
- **Emotion Dynamics**:
  - Analyze emotional tone shifts across dialogues using polarity scores.
  - Combine this with word embedding models for deeper semantic shifts.

---

## 6. Genre-Specific Analysis (``Melis``)
- **TF-IDF and Wordclouds**:
  - Generate genre-specific word clouds using TF-IDF for the top terms in dialogues.
  - Analyze the top emotional and thematic keywords for each genre.

---

## 7. Visualizations
- **Network Visualizations**: (``Johan``)
  - Create clear visual representations for:
    - Actor-movie bipartite graph.
    - Movie-movie collaborations with thresholds.
    - Community structures.
  - Annotate visualizations to emphasize key findings like highly connected nodes (e.g., Samuel L. Jackson).
- **Sentiment Trends**: (``Melis``)
  - Plot average sentiment by genre over decades.
  - Show cumulative emotional arcs for representative movies.

---

## 8. Report Writing
- **Sections to Prioritize**:
  - **Abstract**: Refine based on the final findings.
  - **Methods**: Ensure detailed explained
  - **Results**: Ensure all our results are written neatly and we refer to nicely made plots with correct captioning etc.
  - **Discussion**: Create a great discussion, lacking, good, would have been nice, etc
  - **Table of contribution**: Outline who did what
