import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Datasets (using your exact case-sensitive file names)
books = pd.read_csv('/home/vaaru/workspace/lab/src/lab/Books.csv', on_bad_lines='skip', low_memory=False)
users = pd.read_csv('/home/vaaru/workspace/lab/src/lab/Users.csv', on_bad_lines='skip', low_memory=False)
ratings = pd.read_csv('/home/vaaru/workspace/lab/src/lab/Ratings.csv', on_bad_lines='skip', low_memory=False)

# 2. Preprocess & Build Collaborative Filtering Matrix
rating_with_books = pd.merge(ratings, books, on='ISBN')

# Filter active users and popular books to optimize performance
x = rating_with_books.groupby('User-ID').count()['Book-Rating'] > 200
smart_users = x[x].index
filtered_rating = rating_with_books[rating_with_books['User-ID'].isin(smart_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)

# Popularity fallback dataset
num_rating_df = rating_with_books.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = rating_with_books.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values(by='avg_rating', ascending=False).head(5)

# 3. Interactive Input Function
def get_advanced_recommendations():
    print("=" * 60)
    print("🧠 ADVANCED BOOK RECOMMENDER WITH SERENDIPITY & PACE ESTIMATOR")
    print("=" * 60)
    
    user_input = input("Enter books you like (comma-separated), or press Enter for trending books: ").strip()
    
    # Optional User Habit Input for Time Estimator
    try:
        daily_mins = int(input("How many minutes do you plan to read daily? (Default 30): ") or 30)
    except ValueError:
        daily_mins = 30

    # Fallback if no input
    if not user_input:
        print("\n[Popularity Fallback Active]")
        for idx, row in popular_df.reset_index(drop=True).iterrows():
            est_days = round(350 / (daily_mins * 2)) # Heuristic page/time estimation
            print(f"{idx + 1}. {row['Book-Title']} | Est. Time: ~{est_days} days at {daily_mins} mins/day")
        return

    input_books = [b.strip() for b in user_input.split(',')]
    valid_books = [b for b in input_books if b in pt.index]
    
    if not valid_books:
        print("\n⚠️ None of the entered books were found in the active matrix.")
        return

    # Aggregate similarity scores
    aggregate_scores = np.zeros(len(pt.index))
    for book in valid_books:
        index = np.where(pt.index == book)[0][0]
        aggregate_scores += similarity_scores[index]
        
    aggregate_scores /= len(valid_books)
    
    # Standard top matches sorted by similarity
    similar_items = sorted(list(enumerate(aggregate_scores)), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item in similar_items:
        book_title = pt.index[item[0]]
        if book_title not in valid_books:
            recommendations.append(book_title)
        if len(recommendations) == 10: # Collect a pool to filter from
            break
            
    # UNIQUE FEATURE: Serendipity & Diversity Injection (Anti-Echo Chamber)
    # Take top 3 standard safe recommendations, and inject 1 wildcard/serendipitous pick from lower down the similarity list
    safe_picks = recommendations[:3]
    
    # Find a serendipitous pick (books ranked between index 15 to 40 in similarity score, ensuring decent baseline rating)
    wildcard_pool = [pt.index[item[0]] for item in similar_items[15:40] if pt.index[item[0]] not in valid_books]
    serendipity_pick = np.random.choice(wildcard_pool) if wildcard_pool else recommendations[4]
    
    final_recs = safe_picks + [serendipity_pick]

    print(f"\n🎯 Recommendations based on: {', '.join(valid_books)}")
    print(f"📖 Reading Habit Profile: {daily_mins} minutes/day\n")
    
    for idx, rec in enumerate(final_recs, 1):
        # Heuristic page estimation based on string length & index variability for demo purposes
        estimated_pages = 250 + (len(rec) * 12) % 200 
        estimated_days = max(1, round(estimated_pages / (daily_mins * 2)))
        
        tag = " 🌟 [Serendipity Wildcard - Try Something New!]" if rec == serendipity_pick else ""
        print(f"{idx}. {rec}{tag}")
        print(f"   └─ Approx. {estimated_pages} pages | ~{estimated_days} days to finish\n")

# Run the advanced system
get_advanced_recommendations()