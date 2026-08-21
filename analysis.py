import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/movies.csv")

print("MOVIE DATA ANALYSIS")
print("=" * 40)

# ==========================================
# 2. DATASET INFORMATION
# ==========================================

print("\nDATASET SIZE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# ==========================================
# 3. DATA CLEANING
# ==========================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# ==========================================
# 4. BASIC STATISTICS
# ==========================================

print("\nBASIC STATISTICS:")
print(df.describe())

# ==========================================
# 5. TOP-RATED MOVIES
# ==========================================

top_10 = df.sort_values(
    by="Rating",
    ascending=False
).head(10)

print("\nTOP 10 HIGHEST-RATED MOVIES:")
print(top_10[["Movie_Title", "Genre", "Rating"]])

highest_rated = df.loc[df["Rating"].idxmax()]

print("\nHIGHEST-RATED MOVIE:")
print(highest_rated["Movie_Title"])
print("Rating:", highest_rated["Rating"])

# ==========================================
# 6. GENRE ANALYSIS
# ==========================================

genre_count = df["Genre"].value_counts()

print("\nMOVIES BY GENRE:")
print(genre_count)

print("\nMOST COMMON GENRE:")
print(genre_count.idxmax())

# ==========================================
# 7. RELEASE YEAR ANALYSIS
# ==========================================

movies_per_year = (
    df["Release_Year"]
    .value_counts()
    .sort_index()
)

print("\nMOVIES RELEASED BY YEAR:")
print(movies_per_year)

# ==========================================
# 8. RATING BY GENRE
# ==========================================

average_rating = (
    df.groupby("Genre")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE RATING BY GENRE:")
print(average_rating)

print("\nHIGHEST-RATED GENRE:")
print(average_rating.idxmax())

# ==========================================
# 9. REVENUE ANALYSIS
# ==========================================

highest_revenue = df.loc[df["Revenue"].idxmax()]

print("\nHIGHEST REVENUE MOVIE:")
print(highest_revenue["Movie_Title"])
print("Revenue:", highest_revenue["Revenue"])

top_revenue = df.sort_values(
    by="Revenue",
    ascending=False
).head(10)

print("\nTOP 10 MOVIES BY REVENUE:")
print(top_revenue[["Movie_Title", "Genre", "Revenue"]])

revenue_by_genre = (
    df.groupby("Genre")["Revenue"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE REVENUE BY GENRE:")
print(revenue_by_genre)

# ==========================================
# 10. DIRECTOR ANALYSIS
# ==========================================

director_count = df["Director"].value_counts()

print("\nTOP DIRECTORS BY NUMBER OF MOVIES:")
print(director_count.head(10))

# ==========================================
# 11. RUNTIME ANALYSIS
# ==========================================

print("\nAVERAGE MOVIE RUNTIME:")
print(df["Runtime"].mean())

longest_movie = df.loc[df["Runtime"].idxmax()]

print("\nLONGEST MOVIE:")
print(longest_movie["Movie_Title"])
print("Runtime:", longest_movie["Runtime"])

# ==========================================
# 12. CORRELATION ANALYSIS
# ==========================================

numeric_data = df[
    ["Rating", "Votes", "Revenue", "Runtime"]
]

correlation = numeric_data.corr()

print("\nCORRELATION MATRIX:")
print(correlation)

# ==========================================
# 13. VISUALIZATIONS
# ==========================================

# Movies by Genre
plt.figure(figsize=(10, 6))
genre_count.plot(kind="bar")
plt.title("Number of Movies by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/movies_by_genre.png")
plt.close()

# Movies by Year
plt.figure(figsize=(10, 6))
movies_per_year.plot(kind="line", marker="o")
plt.title("Movies Released by Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/movies_by_year.png")
plt.close()

# Average Rating by Genre
plt.figure(figsize=(10, 6))
average_rating.plot(kind="bar")
plt.title("Average Movie Rating by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/average_rating_by_genre.png")
plt.close()

# Average Revenue by Genre
plt.figure(figsize=(10, 6))
revenue_by_genre.plot(kind="bar")
plt.title("Average Revenue by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/average_revenue_by_genre.png")
plt.close()

# Rating vs Revenue
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Rating",
    y="Revenue"
)
plt.title("Movie Rating vs Revenue")
plt.xlabel("Rating")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("output/rating_vs_revenue.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Between Movie Attributes")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.close()

print("\n========================================")
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("========================================")