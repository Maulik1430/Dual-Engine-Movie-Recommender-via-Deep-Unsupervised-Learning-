# 🎬 MovieLens Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)

**A dual-approach movie recommendation system using collaborative filtering techniques**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure)

</div>

---
## Live Demo
This project is deployed as a Streamlit web application.

👉 [Launch the app]([https://utilization-of-cnn-models-for-animal-species-detection-y8pfnhp.streamlit.app](https://hnkfg4dsyccobvergzxapo.streamlit.app/))
## 📖 Overview

This project implements a **collaborative filtering-based movie recommendation system** using the MovieLens 1M dataset. It provides personalized movie recommendations through two powerful unsupervised learning approaches:

- **🔮 SVD (Singular Value Decomposition)** - Matrix factorization for pattern discovery
- **👥 User-Based KNN** - Finding similar users through demographic and rating patterns

The interactive web interface, built with Streamlit, allows users to explore recommendations side-by-side and understand how different algorithms perceive their preferences.

---

## ✨ Features

### 🎯 Dual Recommendation Engines
- **SVD-based recommendations** using matrix factorization to uncover latent features
- **User similarity-based recommendations** using K-Nearest Neighbors algorithm

### 🎨 Interactive Web Interface
- Clean, modern UI built with Streamlit
- Side-by-side comparison of both recommendation methods
- Real-time parameter tuning for both algorithms

### 📊 Rich User Experience
- View user demographics and watch history
- Star-rating visualization (★★★☆☆)
- Movie posters for visual browsing
- Genre information for each recommendation

### ⚙️ Customizable Parameters
- **SVD Settings**: Adjust number of components (latent factors)
- **User-Based Settings**: 
  - Choose similarity criteria (Age, Gender, Occupation)
  - Control number of similar neighbors
- **General**: Set number of recommendations to display

---

## 🎥 Demo

### User Profile & Watch History
View the selected user's demographics and their previously rated movies with star ratings:

```
User: ID 1234
Gender: M | Age: 25 | Occupation: Student
```

### Recommendations Comparison

| SVD Recommendations | User Similarity Recommendations |
|---------------------|--------------------------------|
| Based on hidden patterns in rating behavior | Based on similar users' preferences |
| Matrix factorization approach | Collaborative filtering approach |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/movielens-recommendation.git
cd movielens-recommendation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
streamlit
pandas
numpy
scipy
scikit-learn
```

### Step 3: Prepare the Dataset
Download the MovieLens 1M dataset and organize it as follows:

```
project-root/
│
├── movie-lens-1m/
│   ├── users.csv
│   ├── movies.csv
│   └── ratings.csv
│
├── Movie_Lens_recomendation.py
├── 25-svd-movielens.ipynb
├── KNN-movielens.ipynb
└── README.md
```

**Dataset Structure:**
- `users.csv`: User demographics (user_id, age, gender, occupation)
- `movies.csv`: Movie information (movie_id, title, genres)
- `ratings.csv`: User ratings (user_id, movie_id, rating, timestamp)

---

## 💻 Usage

### Running the Application
```bash
streamlit run Movie_Lens_recomendation.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Select a User**: Choose a user ID from the sidebar dropdown
2. **Set Recommendations**: Adjust the slider for number of movies to recommend (1-20)
3. **Configure SVD**: 
   - Expand "SVD Settings"
   - Adjust number of components (1-100) - higher values capture more patterns
4. **Configure User-Based**:
   - Expand "User-Based Settings"
   - Select similarity criteria (Age, Gender, Occupation)
   - Set number of neighbors to consider (1-20)
5. **Explore**: Compare recommendations from both approaches!

---

## 🧠 How It Works

### 🔮 SVD (Singular Value Decomposition) Approach

SVD is a matrix factorization technique that discovers hidden patterns in user-movie rating data.

```
Original Rating Matrix → SVD → Reduced Dimensions → Predicted Ratings
```

**Process:**
1. **Create Matrix**: Build a user-movie rating matrix (users × movies)
2. **Normalize**: Subtract each user's average rating to remove bias
3. **Decompose**: Apply SVD to find `k` latent factors
   ```
   R = U × Σ × V^T
   ```
   - **U**: User features in latent space
   - **Σ**: Strength of each latent factor
   - **V^T**: Movie features in latent space
4. **Reconstruct**: Generate predicted ratings for all movies
5. **Recommend**: Select top-K unwatched movies with highest predicted ratings

**Advantages:**
- Captures complex, non-obvious patterns
- Works well with sparse data
- Reduces dimensionality (noise reduction)

---

### 👥 User-Based KNN Approach

Finds similar users and recommends movies they enjoyed.

```
Filter Similar Users → Find K Nearest Neighbors → Aggregate Their Ratings
```

**Process:**
1. **Filter Users**: Select users matching chosen criteria:
   - Same age group
   - Same gender
   - Same occupation
2. **Normalize Ratings**: Subtract each user's average rating
3. **Find Neighbors**: Use KNN with cosine similarity to find K most similar users
4. **Aggregate**: Calculate average ratings from similar users
5. **Recommend**: Select top-K unwatched movies with highest average ratings

**Advantages:**
- Intuitive and explainable
- Leverages demographic similarity
- Adapts to niche preferences

---

## 📁 Project Structure

```
movielens-recommendation/
│
├── 📄 Movie_Lens_recomendation.py    # Main Streamlit application
├── 📓 25-svd-movielens.ipynb         # SVD training & experimentation notebook
├── 📓 KNN-movielens.ipynb            # KNN training & experimentation notebook
│
├── 📂 movie-lens-1m/                 # Dataset directory
│   ├── users.csv                     # User demographics
│   ├── movies.csv                    # Movie catalog
│   └── ratings.csv                   # User ratings
│
├── 📄 README.md                      # Project documentation
└── 📄 requirements.txt               # Python dependencies
```

### 📓 Jupyter Notebooks

The project includes two educational notebooks for understanding the algorithms:

- **`25-svd-movielens.ipynb`**: Deep dive into SVD
  - Matrix factorization theory
  - Parameter tuning experiments
  - Performance analysis
  
- **`KNN-movielens.ipynb`**: User-based filtering exploration
  - Similarity metrics comparison
  - Neighbor selection strategies
  - Demographic impact analysis

---

## 🎨 Credits & Acknowledgments

### Dataset
This project uses the **MovieLens 1M Dataset** provided by GroupLens Research.
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

### Movie Posters
Movie poster images are sourced from **Kaveh Bakhtiyari's** excellent MovieLens poster collection:
- Repository: [kavehbc/movielens-posters](https://github.com/kavehbc/movielens-posters)
- Direct Link: `https://raw.githubusercontent.com/kavehbc/movielens-posters/refs/heads/master/posters/{movie_id}.jpg`

**Thank you, Kaveh!** 🙏 Your poster collection brings the recommendations to life!

---

## 🛠️ Technical Details

### Key Technologies
- **Streamlit**: Interactive web framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Sparse matrix operations for SVD
- **Scikit-learn**: KNN implementation

### Algorithms Implemented

| Algorithm | Type | Complexity | Best For |
|-----------|------|------------|----------|
| SVD | Matrix Factorization | O(mnk) | Discovering hidden patterns |
| User-KNN | Memory-based CF | O(n²) | Explainable recommendations |

### Performance Considerations
- **SVD**: Pre-computed and cached using `@st.cache_data`
- **KNN**: Computed on-demand based on user selection
- **Scalability**: Efficient for MovieLens 1M (1M ratings, 6K users, 4K movies)

---

## 📊 Understanding the Recommendations

### When SVD Excels
- Users with diverse, eclectic taste
- Discovering non-obvious connections
- Cold-start scenarios (few ratings)

### When User-Based KNN Excels
- Strong demographic correlations
- Niche genre preferences
- Trust in similar users' opinions

### Comparing Results
The side-by-side display lets you:
- See where algorithms agree (high confidence)
- Identify unique discoveries from each approach
- Understand your taste profile better

---

## 🔧 Customization

### Adjusting SVD Components
```python
# In the sidebar
n_components = st.slider("# of Components", min_value=1, max_value=100, value=50)
```
- **Lower values (10-30)**: Faster, captures main patterns, may miss nuances
- **Higher values (50-80)**: Slower, captures subtle patterns, risk of overfitting
- **Sweet spot**: Usually 40-60 for MovieLens 1M

### Fine-tuning User Similarity
```python
# Choose criteria that matter to you
similarity_criteria = ["Age", "Gender", "Occupation"]
no_of_neighbors = 5  # Number of similar users to consider
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Add hybrid recommendation approach (SVD + KNN)
- Implement content-based filtering using genres
- Add evaluation metrics (RMSE, Precision@K)
- Create visualization of latent factors
- Add explanation for each recommendation

### Steps to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 FAQ

**Q: Why are some movies showing broken poster images?**  
A: Not all movies in MovieLens 1M have posters in Kaveh's collection. This is normal.

**Q: Can I use my own dataset?**  
A: Yes! Just ensure your CSVs match the expected format (user_id, movie_id, rating, etc.)

**Q: Which algorithm should I trust more?**  
A: Neither is "better" - they complement each other! SVD finds hidden patterns, while KNN provides explainable, intuitive recommendations.

**Q: Why do recommendations change when I adjust parameters?**  
A: This is expected! You're exploring different perspectives:
- SVD components = different granularity of patterns
- Similarity criteria = different peer groups
- Number of neighbors = broader or narrower consensus

**Q: How long does SVD computation take?**  
A: First run may take 10-30 seconds. After that, it's cached and instant!

---

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Reach out to the maintainer

---

## 🌟 Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **Kaveh Bakhtiyari** for the movie poster collection
- **Streamlit team** for the amazing framework
- **Open source community** for inspiration and tools

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

Made with ❤️ and Python

</div>
