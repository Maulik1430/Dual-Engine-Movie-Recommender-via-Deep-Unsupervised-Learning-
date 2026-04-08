import streamlit as st
import pandas as pd
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors

def convert_rating_to_stars(rating):
    """Convert a rating to a star representation."""
    full_stars = int(rating)
    half_star = 1 if rating % 1 >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    return "★" * full_stars + "½" * half_star + "☆" * empty_stars

def calculate_similarity_recommendations(df_ratings, df_user, n_neighbors=5, similarity_criteria=["Age"]):
    df_users = pd.read_csv("movie-lens-1m/users.csv")

    if "Age" in similarity_criteria:
        user_age = df_user["age"].values[0]
        df_users = df_users[df_users["age"] == user_age]
    if "Gender" in similarity_criteria:
        user_gender = df_user["gender"].values[0]
        df_users = df_users[df_users["gender"] == user_gender]
    if "Occupation" in similarity_criteria:
        user_occupation = df_user["occupation"].values[0]
        df_users = df_users[df_users["occupation"] == user_occupation]

    df_users = df_users.reset_index(drop=True)
    
    # filter ratings to only include users similar to the target user
    df_ratings = df_ratings[df_ratings["user_id"].isin(df_users["user_id"])]
    
    df_pivot = df_ratings.pivot(index="user_id", columns="movie_id", values="rating")
    
    users_avg_rating = df_pivot.mean(axis=1)
    df_norm = df_pivot.sub(users_avg_rating, axis='rows')
    df_norm = df_norm.fillna(0)
    norm_matrix = df_norm.to_numpy()

    # use KNN to find similar users
    model = NearestNeighbors(n_neighbors=n_neighbors+1, metric='cosine')
    model.fit(norm_matrix)
    distances, indices = model.kneighbors(norm_matrix)
    target_user_index = df_pivot.index.get_loc(df_user['user_id'].values[0])
    similar_user_indices = indices[target_user_index][1:]  # exclude the target user itself
    similar_user_ids = df_pivot.index[similar_user_indices]
    df_similar_ratings = df_ratings[df_ratings["user_id"].isin(similar_user_ids)]
    # st.write(similar_user_ids)
    df_similar_ratings = df_similar_ratings.groupby("movie_id")["rating"].mean().reset_index()
    # st.write(df_similar_ratings)
    
    df_predicted = df_similar_ratings

    return df_predicted

@st.cache_data(persist=True)
def calculate_svd_recommendations(df_ratings, k=50):
    df_pivot = df_ratings.pivot(index="user_id", columns="movie_id", values="rating")
    users_avg_rating = df_pivot.mean(axis=1)
    df_norm = df_pivot.sub(users_avg_rating, axis='rows')
    df_norm = df_norm.fillna(0)
    norm_matrix = df_norm.to_numpy()
    u, s, vt = scipy.sparse.linalg.svds(norm_matrix, k=k)

    # reconstruct the matrix
    regenerated = np.dot(u, np.dot(np.diag(s), vt))
    
    df_predicted = pd.DataFrame(data=regenerated,
                            columns=df_pivot.columns,
                            index=df_pivot.index)
    df_predicted = df_predicted.add(users_avg_rating, axis='rows')

    return df_predicted

def main():
    st.title("MovieLens Movie Recommendation (Collaborative Filtering)")
    df_users = pd.read_csv("movie-lens-1m/users.csv")
    df_movies = pd.read_csv("movie-lens-1m/movies.csv")
    
    df_movies['thumbnail'] = df_movies['movie_id'].apply(
        lambda x: f"https://raw.githubusercontent.com/kavehbc/movielens-posters/refs/heads/master/posters/{x}.jpg"
    )
    
    df_ratings = pd.read_csv("movie-lens-1m/ratings.csv")
    df_ratings_movies = pd.merge(df_ratings, df_movies, on='movie_id')

    with st.sidebar:
        user_id = st.selectbox("Select a user ID", df_users['user_id'].unique())
        K = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

        with st.expander("SVD Settings"):
            n_components = st.slider("\# of Components", min_value=1, max_value=100, value=50)

        with st.expander("User-Based Settings"):
            similarity_criteria = st.multiselect("Select similarity criteria", ["Age", "Gender", "Occupation"])
            no_of_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5)


    st.subheader("User Information")
    df_user = df_users[df_users['user_id'] == user_id]
    st.write(df_user)
    
    st.subheader("User Ratings")

    df_watched_movies = df_ratings_movies[df_ratings_movies['user_id'] == user_id][["thumbnail", "title", "genres", "rating"]].reset_index(drop=True)

    df_watched_movies["rating"] = df_watched_movies["rating"].apply(lambda x: convert_rating_to_stars(x))

    st.dataframe(df_watched_movies,
                 column_config={"thumbnail": st.column_config.ImageColumn("Movie Poster"),
                                # "rating": st.column_config.ProgressColumn(
                                # "Rating",  # Label for the column
                                # help="This column shows the progress of each task",
                                # format="",  # Format the display value as a percentage
                                # min_value=1,
                                # max_value=5,
                                # )
                                },
                            hide_index=True)


    user_watched_movies = df_ratings_movies[df_ratings_movies['user_id'] == user_id]['movie_id'].tolist()
    
    st.subheader("Recommended Movies")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SVD")

        df_predicted = calculate_svd_recommendations(df_ratings, k=n_components)
        df_user_prediction = df_predicted[df_predicted.index == user_id]
        df_user_prediction = df_user_prediction.T.reset_index().sort_values(by=[user_id], ascending=False)

        n = 0
        recommendations = []
        for _, row in df_user_prediction.iterrows():
            if row["movie_id"] not in user_watched_movies:
                recommendations.append(int(row["movie_id"]))
                n += 1
                if n == K:
                    break

        df_movies_recommendation = df_movies[df_movies["movie_id"].isin(recommendations)]

        st.dataframe(df_movies_recommendation[["thumbnail", "title", "genres"]],
                    column_config={"thumbnail": st.column_config.ImageColumn("Movie Poster")},
                    hide_index=True)
        
    with col2:
        st.subheader("User's Similarity")
        df_predicted = calculate_similarity_recommendations(df_ratings, df_user, n_neighbors=no_of_neighbors, similarity_criteria=similarity_criteria)
        df_predicted = df_predicted.sort_values(by=["rating"], ascending=False)

        n = 0
        recommendations = []
        for _, row in df_predicted.iterrows():
            if row["movie_id"] not in user_watched_movies:
                recommendations.append(int(row["movie_id"]))
                n += 1
                if n == K:
                    break

        df_movies_recommendation = df_movies[df_movies["movie_id"].isin(recommendations)]

        st.dataframe(df_movies_recommendation[["thumbnail", "title", "genres"]],
                    column_config={"thumbnail": st.column_config.ImageColumn("Movie Poster")},
                    hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="MovieLens Movie Recommendation",
                       layout="wide")
    main()
