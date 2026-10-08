import json
import string

def keyword_search(search_query:str, json_path = "/home/ppappas/rag-search-engine/data/movies.json"):
    movies_dic = {}
    result = []
    with open(json_path, 'r', encoding= 'utf-8') as file:
        movies_dic = json.load(file)
    for movie in movies_dic["movies"]:
        title :str = movie["title"]
        title_lowered: str = movie["title"].lower()
        title_lowered = title_lowered.translate(str.maketrans(string.punctuation,string.punctuation,string.punctuation))
        print(f"these are the raw titles :{title}")
        print(f"this is the searched_titles: {title_lowered}")
        if search_query.translate(str.maketrans(string.punctuation,string.punctuation,string.punctuation)) in title_lowered:
                print(f"this is the searched string: {search_query}")
                result.append(title)
    x=1
    for title in result:
         print(f"{title}")
         x+=1

