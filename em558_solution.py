#Importing the necessary libraries for all the tasks below
import requests
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup

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

#Task 1

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

'''
Identifying where there may be missing entries or null values, in this case all the numerical features,
in this case all the numerical features have no missing entries so no cleaning will be done, however,
with the categorical features, Author and Publisher are missing values (below 21100), so the values will be ultimately dropped
from Author given that it won't remove too much of the data, and missing values in Publisher will be replaced with 'Unknown',
given that it would remove too much of the data if dropped albeit it's less of an important feature for the recommender engine.
''' 


#Dropping missing values in Author Column
combined_df.dropna(subset=['Author'], inplace=True)


combined_df['Publisher'] = combined_df['Publisher'].fillna('Unknown')


print(combined_df.isnull().sum())
print(combined_df.shape)

'''
After running describe before cleaning and after cleaning the data the values remain almost the same, 
even though 2400 entries were dropped so, so far the data still seems reliable to use for the reccomender engine.
'''
print(combined_df.describe())
print(combined_df.describe(include='str'))


#Task 2

#Calculating the mean rating for each book given its title and printing the top 10 highest rated books
average_ratings = combined_df.groupby('Title')['rating'].mean().sort_values(ascending=False)
print(average_ratings.head(10))

bootstrap = []
for i in range(1000):
    sample = combined_df['rating'].sample(n=100, replace=True)
    bootstrap.append(sample.mean())


lower = np.percentile(bootstrap, 2.5)
upper = np.percentile(bootstrap, 97.5)
print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")