#Importing the necessary libraries for all the tasks below
import requests
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors

#Part 1: Web scraping using BeautifulSoup
def collect_page_data(url, csv_filename='BBCrecipe.csv'):
    """
    Scrapes a BBC Food recipe page and returns a dataframe with key info
    The info also gets saved to an external CSV file. 
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return pd.DataFrame()

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract structured recipe data from the JSON-LD script tag
    script_tag = soup.find('script', type='application/ld+json')
    if not script_tag:
        print("No data found on this page.")
        return pd.DataFrame()

    #Exception handling if JSON doesn't return in a valid format
    try:
        raw = json.loads(script_tag.string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON data: {e}")
        return pd.DataFrame()

    # Some pages nest the recipe inside an @graph array
    if '@graph' in raw:
        recipe_items = [item for item in raw['@graph'] if item.get('@type') == 'Recipe']
        if not recipe_items:
            print("No Recipe found.")
            return pd.DataFrame()
        data = recipe_items[0]
    else:
        data = raw

    # Extract title
    title = data.get('name') or None

    # Extract image URL (can be a list, dict, or string)
    img = data.get('image')
    if isinstance(img, list):
        image = img[0].get('url') if img else None
    elif isinstance(img, dict):
        image = img.get('url') or None
    else:
        image = img or None

    # Extract prep and cook times from the HTML dt/dd pairs
    prep_time = None
    cook_time = None
    for dt in soup.find_all('dt'):
        label = dt.text.strip().lower()
        dd = dt.find_next_sibling('dd')
        if dd:
            if label == 'prepare':
                prep_time = dd.text.strip()
            elif label == 'cook':
                cook_time = dd.text.strip()
    total_time = f"Prep: {prep_time}, Cook: {cook_time}" if prep_time or cook_time else None

    # Extract category, cuisine, and ingredients
    category = data.get('recipeCategory') or None
    cuisine = data.get('recipeCuisine') or None
    ingredient_list = data.get('recipeIngredient', [])
    ingredients = ', '.join(ingredient_list) if ingredient_list else None

    # Extract rating information
    agg_rating = data.get('aggregateRating', {})
    rating_val = agg_rating.get('ratingValue') or None
    rating_count = agg_rating.get('ratingCount') or None

    # Extract dietary information and derive vegan/vegetarian flags
    diets = data.get('suitableForDiet', [])
    if isinstance(diets, str):
        diets = [diets]
    diet_str = ', '.join(diets) if diets else None
    vegan = any('Vegan' in d for d in diets) if diets else None
    vegetarian = any('Vegetarian' in d for d in diets) if diets else None

    # Build DataFrame and save to CSV
    columns = {
        'title': title,
        'total_time': total_time,
        'image': image,
        'ingredients': ingredients,
        'rating_val': rating_val,
        'rating_count': rating_count,
        'category': category,
        'cuisine': cuisine,
        'diet': diet_str,
        'vegan': vegan,
        'vegetarian': vegetarian,
        'url': url
    }
    df = pd.DataFrame([columns])
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig', na_rep='NaN')
    return df

# Test with 3 different BBC recipe pages
url1 = "https://www.bbc.co.uk/food/recipes/easiest_ever_banana_cake_42108"
url2 = "https://www.bbc.co.uk/food/recipes/dijon_mustard_roast_23740"
url3 = "https://www.bbc.co.uk/food/recipes/tomato_soup_56817"

df1 = collect_page_data(url1, 'BBCrecipe1.csv')
print(df1)
df2 = collect_page_data(url2, 'BBCrecipe2.csv')
print(df2)
df3 = collect_page_data(url3, 'BBCrecipe3.csv')
print(df3)

#Part 2 (Guided): Building Up a recommender engine

#Part 2, Task 1

#Reading the csv files using pandas which also converts them to dataframes.
books_df = pd.read_csv('books_new.csv')
ratings_df = pd.read_csv('ratings.csv')

#Merging the two dataframes on the shared column 'bookId'.
combined_df = pd.merge(ratings_df, books_df, on='bookId')

#Identifying the numerical and categorical feature columns from both dataframes.

numerical = combined_df.select_dtypes(include='number').columns.tolist()
categorical = combined_df.select_dtypes(include='str').columns.tolist()

print("Numerical features:", numerical)
print("Categorical features:", categorical)


#Identifying where there may be missing entries or null values, in this case all the numerical features,
#in this case all the numerical features have no missing entries so no cleaning will be done, however,
#with the categorical features, Author and Publisher are missing values (below 21100), so the values will be ultimately dropped
#from Author given that it won't remove too much of the data, and missing values in Publisher will be replaced with 'Unknown',
#given that it would remove too much of the data if dropped albeit it's less of an important feature for the recommender engine.



#Dropping missing values in Author Column
combined_df.dropna(subset=['Author'], inplace=True)


combined_df['Publisher'] = combined_df['Publisher'].fillna('Unknown')


print(combined_df.isnull().sum())
print(combined_df.shape)


#After running describe before cleaning and after cleaning the data the values remain almost the same, 
#even though 2400 entries were dropped so, so far the data still seems reliable to use for the reccomender engine.

print(combined_df.describe())
print(combined_df.describe(include='str'))


#Part 2, Task 2

#Calculating the mean rating for each book given its title and printing the top 10 highest rated books
average_ratings = combined_df.groupby('Title')['rating'].mean().sort_values(ascending=False)
print(average_ratings.head(10))


#Using Bootstrapping method to compute a 95% confidence interval for the average (mean),
#by creating 1000 samples of size 100 with replacement. 

bootstrap = []
for i in range(1000):
    sample = combined_df['rating'].sample(n=100, replace=True)
    bootstrap.append(sample.mean())


lower = np.percentile(bootstrap, 2.5)
upper = np.percentile(bootstrap, 97.5)
print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")


#Part 2, Task 3

#Adding an average rating column and rating count column into existing DF.
average_ratings = average_ratings.to_frame(name='average_rating')
average_ratings['rating_count'] = combined_df.groupby('Title')['rating'].count()
print(average_ratings.head(10),"\n")

average_ratings.plot.scatter(x='rating_count', y='average_rating')
plt.title('Average Rating vs Rating Count')
plt.xlabel('Rating Count')
plt.ylabel('Average Rating')
plt.show()


#All books in this dataset have 100 ratings each so there's no relationship between the average rating
#and the number of ratings, for example you don't see a trend where, 
#a book with a higher average rating doesn't show that the rating count is smaller,
#therefore the average is more extreme. 


#Part 2, Task 4 a.

#Implementing the user liking or disliking a book based on the thershold being 3.6, 1 represents like and -1 dislike.
combined_df['rating'] = np.where(combined_df['rating'] >= 3.6, 1, -1)


#Creating a new DF which drops duplicate entries of each novel title, otherwise after making the recommender,
#the most similar novels to a given novel will be the novel itself.

books_df = combined_df.drop_duplicates(subset='Title').reset_index(drop=True)

#Identifying the features of the DF in separate variable so that each column of a book can be combined into a single string. 
features = ['Title', 'Author', 'Genre', 'SubGenre', 'Publisher']
books_df['combined_features'] = books_df[features].agg(' '.join, axis=1)

#Importing CountVectorizer method from SkLearn which will convert the combined features column into a matrix
#cosine_sim uses the cosine_similarity method to compute the matrix.
cv = CountVectorizer()
count_matrix = cv.fit_transform(books_df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

#Getting the location (index) of where the book called Orientalism is in the titleframe
book_pos = books_df[books_df['Title'] == 'Orientalism'].index[0]

#Grabbing the row from the similarity matrix for the book 'Orientalism'
#which is an array of scores representing how similar each other book is to it,
#then converting that row into a pandas series with the corresponding row indexes of other books 
#so that the similarity scores can all be sorted into the top 10 most similar, in this case to 'Orientalism'.

similarity_scores = pd.Series(cosine_sim[book_pos], index=books_df.index)
top_10 = similarity_scores.sort_values(ascending=False).iloc[1:11]

print("Top 10 recommendations for 'Orientalism':\n")
for index, score in top_10.items():
    print(f"{books_df.loc[index, 'Title']} - Similarity Score: {score:.4f}")
print("\n")




#Part 3 (Open-ended): Building up and to evaluate a recommender engine

#Identifying categorical features, title not included as it doesn't provide similarity to other books, a title doesn't describe the novel.
categorical_features = ['Author', 'Genre', 'SubGenre', 'Publisher']

dummies = pd.get_dummies(books_df[categorical_features])


#Normalising height feature to be between 0 and 1 to match the rest of the features already converted into binary form, 
#before it was 160mm to 283mm, if it were to stay that way height would have a lot more influence on the similarity score
#than the other features making this unfair. It's also unlikely you'd be reccomended a book given its height.

height_norm = (books_df['Height'] - books_df['Height'].min()) / (books_df['Height'].max() - books_df['Height'].min())

feature_matrix = pd.concat([dummies, height_norm], axis=1).astype(float).values


#Task 1 vector space method for similarity 
def vec_space_method(book_title, books_df, feature_matrix):
    
    book_index = books_df[books_df['Title'] == book_title].index[0]
    book_vector = feature_matrix[book_index, :]    

#Manually computing, cosine_similarity(A, B) = dot(A, B) / (||A|| * ||B||)

    scores = feature_matrix @ book_vector
    norms = LA.norm(feature_matrix, axis=1) * LA.norm(book_vector)
    similarities = scores / norms

    similarities_series = pd.Series(similarities, index=books_df.index)
    top_10 = similarities_series.sort_values(ascending=False).iloc[1:11]

    print(f"Top 10 recommendations for '{book_title}':\n")
    for index, score in top_10.items():
        print(f"{books_df.loc[index, 'Title']} - Similarity: {score:.4f}")

    return top_10

vec_space_method('Orientalism', books_df, feature_matrix)


#Task 2 K nearest neighbour (KNN) method for similarity

def knn_similarity(book_title, books_df, feature_matrix):
    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(feature_matrix)

    book_index = books_df[books_df['Title'] == book_title].index[0]
    distances, indices = knn.kneighbors([feature_matrix[book_index]])

    print(f"Top 10 recommendations for '{book_title}':\n")
    for i in range(1, 11):
        title = books_df.loc[books_df.index[indices[0][i]], 'Title']
        similarity = 1 - distances[0][i]  # convert distance to similarity
        print(f"{title} - Similarity: {similarity:.4f}")

    return indices[0][1:11]

knn_similarity('Orientalism', books_df, feature_matrix)


#Given the results and the code above, both methods use cosine similarity to 
#calculate the similarity between each book using the same feature matrix,
#therefore they produce the same recommendation scores.


#Task 3, Evaluating both recommender systems

test_set = {
    'User 1': 'Fundamentals of Wavelets',
    'User 2': 'Orientalism',
    'User 3': 'How to Think Like Sherlock Holmes',
    'User 4': 'Data Scientists at Work'
}

total_books = len(books_df)

space_vector_recommendations = {}
knn_recommendations = {}

for user, book in test_set.items():

    #Spacing out results for each given user as it was too cluttered before
    print(f"\n{'-'*40}")
    print(f"{user} likes '{book}'")

    print("\nVector Space Method:")
    vec_top10 = vec_space_method(book, books_df, feature_matrix)
    space_vector_recommendations[user] = vec_top10.index.tolist()
    
    print("\nKNN Method:")
    knn_top10 = knn_similarity(book, books_df, feature_matrix)
    knn_recommendations[user] = knn_top10.tolist()
    print(f"\n{'-'*40}")

#Both methods produce identical recommendations as they use the same
#feature matrix and cosine similarity method to calculate recommendations. 
#So this part could just be computated once and achieve the same result.
#However, in some instances you see slight differences in how each algorithm
#handles a tie break, for example User 3 has a tie between A Russian Journal,
#and Once there was a war, but Vector space method has A Russian Journal as
#the 8th most similar and KNN has it at 9.

#Evaluating via. coverage which is the unique number of books recommended
#divided by the total number of books per user, in this case 4 test users. 

#Using a set to avoid duplicates, so I can have the unique number of books recommended.
all_recommended_books = set()
for recommendations in space_vector_recommendations.values():
    all_recommended_books.update(recommendations)
coverage = len(all_recommended_books) / total_books

print(f"Coverage of the recommender system: {coverage:.4f}")

#I computed the coverage only once given my previous comment of both vector space method and KNN
#essentially giving the same output.

#Coverage is 0.1667 meaning only 35 out of 210 books were recommended.
#This is expected given that each test user has only been given 10 book recommendations.
#Best case scenario: 4 users * 10 unique books per user = 40/210. 


#Evaluating via. personalisation

#Creating a binary vector for each user 
#representing which books were recommended to them.
#1 if book was recommended to that user, 0 if not.

recommendations_vectors = []
for user in test_set:
    vector = np.zeros(total_books)
    for index in space_vector_recommendations[user]:
        vector[index] = 1
    recommendations_vectors.append(vector)

#Computing the similarity matrix and the  average, A, 
#of the upper triangular matrix entries

similarity_matrix = cosine_similarity(recommendations_vectors)

#4 given that it's 4 users, k=1 to exclude the diagonal of 1s which represent the similarity of a user to themselves.
upper_triangular = similarity_matrix[np.triu_indices(4, k=1)]

#If I used np.triu I'd be returned a 4x4 matrix with everything 
#below the diagonal set to 0 which would mess up the average
#as the 0s are included, instead _indices only gives the values of the
#upper triangular entries.

A = upper_triangular.mean()
personalisation = 1 - A
print(f"Personalisation of the recommender system: {personalisation:.4f}")

#Personalisation score is 0.85 meaning that 
#each test user is getting mostly different books recommended to them
#with a small overlap of 0.1667, this makes sense as the genres of the
#4 test users vary, so you'd expect them to have different recommendations


#Task 4, predictor that predicts whether a user will like a book or not.

#Implementation will be similar to Part 3 Task 1,
#using the vector space method to predict whether a user
#will like a book instead of finding similar books to a given book.

def predict_like(user_id, book_title, books_df, combined_df, feature_matrix):

    #Getting a user rating as a preference vector, 1 for like and -1 for dislike.
    user_ratings = combined_df[combined_df['user_id'] == user_id]

    preference_vector = np.zeros(len(books_df))
    for index, row in user_ratings.iterrows():
        book_index = books_df[books_df['Title'] == row['Title']].index
        if len(book_index) > 0:
            preference_vector[book_index[0]] = row['rating']
        
    
    user_profile = preference_vector @ feature_matrix
#Normalising user profile vector so that a user who has rated a lot more books,
#doesn't have an unfair advantage
    normalisation = LA.norm(user_profile)
    relative_importance = user_profile / normalisation

    weighted_score = feature_matrix @ relative_importance

    book_index = books_df[books_df['Title'] == book_title].index[0]


    if weighted_score[book_index] > 0:
        print(f"Prediction: User {user_id} would LIKE '{book_title}' (score: {weighted_score[book_index]:.4f})")
        return 1
    else:
        print(f"Prediction: User {user_id} would DISLIKE '{book_title}' (score: {weighted_score[book_index]:.4f})")
        return -1

#sample_users = combined_df['user_id'].drop_duplicates().sample(n=10)

#correct = 0
#total = 0

#for user_id in sample_users:
    #user_books = combined_df[combined_df['user_id'] == user_id]
    #for index, row in user_books.iterrows():
        #prediction = predict_like(user_id, row['Title'], books_df, combined_df, feature_matrix)
        #if prediction == row['rating']:
            #correct += 1
        #total += 1

#print(f"\nOverall Accuracy: {correct}/{total} ({correct/total:.4f})")

#Code above used to test accuracy of the predictor.
#After running the test on 10 random users multiple times,
#the accuracy seems to be around 60-90%.




