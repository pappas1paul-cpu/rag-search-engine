import json

def keyword_search(search_query:str, json_path = "/home/ppappas/rag-search-engine/data/movies.json"):
    movies_dic = {}
    result = []
    with open(json_path, 'r', encoding= 'utf-8') as file:
        movies_dic = json.load(file)
    for movie in movies_dic["movies"]:
        title = movie["title"]
        title_lowered = movie["title"].lower()
        if search_query in title_lowered:
                result.append(title)
    x=1
    for title in result:
         print(f"{title}")
         x+=1
