import requests
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import pandas as pd


model = Word2Vec.load('item2vec_model')
word_vectors = model.wv
# del model
df_movies = pd.read_csv('data/movies.csv', error_bad_lines=False, encoding="utf-8")
df_ratings = pd.read_csv('data/ratings.csv')

movieId_to_name = pd.Series(df_movies.title.values, index=df_movies.movieId.values).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index=df_movies.title).to_dict()


def refine_search(search_term):
    """
    Refine the movie name to be recognized by the recommender
    Args:
        search_term (string): Search Term

    Returns:
        refined_term (string): a name that can be search in the dataset
    """
    target_url = "http://www.imdb.com/find?ref_=nv_sr_fn&q="+"+".join(search_term.split())+"&s=tt"
    html = requests.get(target_url).content
    parsed_html = BeautifulSoup(html, 'html.parser')
    for tag in parsed_html.find_all('td', class_="result_text"):
        search_result = re.findall('fn_tt_tt_1">(.*)</a>(.*)</td>', str(tag))
        if search_result:
            if search_result[0][0].split()[0] == "The":
                str_frac = " ".join(search_result[0][0].split()[1:])+", "+search_result[0][0].split()[0]
                refined_name = str_frac+" "+search_result[0][1].strip()
        else:
            refined_name = search_result[0][0] + " " + search_result[0][1].strip()
    return refined_name


def produce_list_of_movieId(list_of_movieName, useRefineSearch=False):
    """
    Turn a list of movie name into a list of movie ids. The movie names has to be exactly the same as they are in the dataset.
       Ambiguous movie names can be supplied if useRefineSearch is set to True

    Args:
        list_of_movieName (List): A list of movie names.
        useRefineSearch (boolean): Ambiguous movie names can be supplied if useRefineSearch is set to True

    Returns:
        list_of_movie_id (List of strings): A list of movie ids.
    """
    list_of_movie_id = []
    for movieName in list_of_movieName:
        if useRefineSearch:
            movieName = refine_search(movieName)
            print("Refined Name: " + movieName)
        if movieName in name_to_movieId.keys():
            list_of_movie_id.append(str(name_to_movieId[movieName]))
    return list_of_movie_id


def recommender(positive_list=None, negative_list=None, useRefineSearch=False, topn=20):
    recommend_movie_ls = []
    if positive_list:
        positive_list = produce_list_of_movieId(positive_list, useRefineSearch)
    if negative_list:
        negative_list = produce_list_of_movieId(negative_list, useRefineSearch)
    for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls


# ls = recommender(positive_list=["Waiting to Exhale"], useRefineSearch=True, topn=5)
ls = recommender(positive_list=["Waiting to Exhale (1995)"], useRefineSearch=False, topn=5)
print('Recommendation Result based on "Waiting to Exhale (1995)":')
print(df_movies[df_movies['movieId'].isin(ls)])

# ls = recommender(positive_list=["Heat (1995)"], negative_list=["Sudden Death (1995)"], useRefineSearch=True, topn=7)
ls = recommender(positive_list=["Heat (1995)"], negative_list=["Sudden Death (1995)"], useRefineSearch=False, topn=7)
print('Recommendation Result based on "Heat (1995)" minus "Sudden Death (1995)":')
print(df_movies[df_movies['movieId'].isin(ls)])

# ls = recommender(positive_list=["Broken Arrow (1996)", "City Hall (1996)"], useRefineSearch=True, topn=7)
ls = recommender(positive_list=["Broken Arrow (1996)", "City Hall (1996)"], useRefineSearch=False, topn=7)
print('Recommendation Result based on "Broken Arrow (1996)" + ""City Hall (1996)":')
print(df_movies[df_movies['movieId'].isin(ls)])
