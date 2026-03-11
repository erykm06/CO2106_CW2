import requests
import pandas as pd
import json
from bs4 import BeautifulSoup

#Part 1: Web scraping using BeautifulSoup

def collect_page_data(url, csv_filename='BBCrecipe.csv'):
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    
    script_tag = soup.find('script', type='application/ld+json')
    raw = json.loads(script_tag.string)
    
    if '@graph' in raw:
        data = next(item for item in raw['@graph'] if item.get('@type') == 'Recipe')
    else:
        data = raw

    title = data.get('name') or None
    img = data.get('image')
    if isinstance(img, list):
        image = img[0].get('url') if img else None
    elif isinstance(img, dict):
        image = img.get('url') or None
    else:
        image = img or None
    # Scrape total time from HTML dt/dd elements
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

    category = data.get('recipeCategory') or None
    cuisine = data.get('recipeCuisine') or None

    ingredient_list = data.get('recipeIngredient', [])
    ingredients = ', '.join(ingredient_list) if ingredient_list else None

    agg_rating = data.get('aggregateRating', {})
    rating_val = agg_rating.get('ratingValue') or None
    rating_count = agg_rating.get('ratingCount') or None

    diets = data.get('suitableForDiet', [])
    if isinstance(diets, str):
        diets = [diets]
    diet_str = ', '.join(diets) if diets else None
    vegan = any('Vegan' in d for d in diets) if diets else None
    vegetarian = any('Vegetarian' in d for d in diets) if diets else None

    row = {
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

    df = pd.DataFrame([row])
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig', na_rep='NaN')
    return df


url1 = "https://www.bbc.co.uk/food/recipes/easiest_ever_banana_cake_42108"
url2 = "https://www.bbc.co.uk/food/recipes/dijon_mustard_roast_23740"
url3 = "https://www.bbc.co.uk/food/recipes/tomato_soup_56817"

df1 = collect_page_data(url1, 'BBCrecipe1.csv')
print(df1)

df2 = collect_page_data(url2, 'BBCrecipe2.csv')
print(df2)

df3 = collect_page_data(url3, 'BBCrecipe3.csv')
print(df3)