import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Select relevant features
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'popularity']

# Handle missing values
df = df.dropna(subset=features)

# Standardize the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Train the K-Means model
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(df[features])

# Save the clustered data for recommendations
df.to_csv('clustered_spotify_tracks.csv', index=False)

# Load the clustered dataset
df = pd.read_csv('clustered_spotify_tracks.csv')

# Function to recommend songs
def recommend_songs(track_name, df):
    track_name = track_name.lower()
    if track_name not in df['track_name'].str.lower().values:
        return "Track not found."
    
    cluster = df.loc[df['track_name'].str.lower() == track_name, 'cluster'].values[0]
    recommended_songs = df[df['cluster'] == cluster]
    return recommended_songs[['track_name', 'artists', 'album_name', 'track_genre']]

# Streamlit UI for interactive queries
st.title('Soundalike Track Recommendation System')

track_name_input = st.text_input('Enter Track Name')
if st.button('Get Recommendations'):
    recommendations = recommend_songs(track_name_input, df)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(recommendations)

# Create sidebar for additional functionality
option = st.sidebar.radio(
    'Select a page:',
    ['Home', 'Recommended Songs Visualization', 'Data Analysis and Model Evaluation']
)

if option == 'Data Analysis and Model Evaluation':
    # Check if the track name exists
    if track_name_input.lower() in df['track_name'].str.lower().values:
        # Get the cluster of the input track
        cluster = df.loc[df['track_name'].str.lower() == track_name_input.lower(), 'cluster'].values[0]

        # Filter the dataframe for the cluster of the input track
        cluster_df = df[df['cluster'] == cluster]

        # Descriptive statistics
        st.header('Data Overview')
        st.write(cluster_df.describe())

        # Monitoring and Maintenance
        if st.checkbox('Show Raw Data'):
            st.subheader('Raw Data')
            st.write(cluster_df)

        # Scatter plot of overall data with highlighted selected song
        st.header('Scatter Plot')
        st.write("Select the features for the X-axis and Y-axis to see a scatter plot of the entire dataset. The selected song will be highlighted in red.")
        x_axis = st.selectbox('Select X-axis', features, index=0)
        y_axis = st.selectbox('Select Y-axis', features, index=1)
        
        selected_song = df[df['track_name'].str.lower() == track_name_input.lower()]
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax, alpha=0.6, edgecolor=None)
        sns.scatterplot(x=x_axis, y=y_axis, data=selected_song, ax=ax, color='red', s=100, label='Selected Song')
        plt.legend()
        st.pyplot(fig)

        # Logging interactions (for monitoring purposes)
        with open('user_interactions.log', 'a') as f:
            f.write(f'{track_name_input} was queried.\n')
    else:
        st.write("Track name not found in the dataset.")

    # Silhouette Score Section
    st.header("Silhouette Score")
    st.write("The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.")

    if st.button('Calculate Silhouette Score'):
        unique_clusters = df['cluster'].nunique()  # Checking the overall dataframe for unique clusters
        if unique_clusters > 1:
            silhouette_avg = silhouette_score(df[features], df['cluster'])
            st.write(f"Silhouette Score: {silhouette_avg}")
        else:
            st.write("Silhouette score cannot be calculated with less than 2 clusters.")

elif option == 'Recommended Songs Visualization':
    # Get recommendations if the track name exists
    if track_name_input.lower() in df['track_name'].str.lower().values:
        recommendations = recommend_songs(track_name_input, df)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.header('Recommended Songs')
            st.write(recommendations)
            
            # Pie chart of genres of recommended songs
            st.header('Genres of Recommended Songs')
            st.write("This pie chart shows the distribution of genres among the top 500 recommended songs.")
            top_500_recommendations = recommendations.head(500)
            fig, ax = plt.subplots()
            top_500_recommendations['track_genre'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            plt.ylabel('')
            st.pyplot(fig)

            # Graph comparing the selected song's popularity against other songs in its genre
            st.header('Popularity of Selected Song vs. Other Songs in Genre')
            st.write("This box plot compares the popularity of the selected song against other songs in the same genre. The selected song is highlighted in red.")
            genre = df.loc[df['track_name'].str.lower() == track_name_input.lower(), 'track_genre'].values[0]
            genre_df = df[df['track_genre'] == genre]
            selected_song_popularity = df.loc[df['track_name'].str.lower() == track_name_input.lower(), 'popularity'].values[0]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(y='track_genre', x='popularity', data=genre_df, ax=ax)
            plt.scatter([selected_song_popularity], [0], color='red', zorder=5, label=f'Selected Song: {track_name_input}')
            plt.legend()
            st.pyplot(fig)
    else:
        st.write("Track name not found in the dataset.")