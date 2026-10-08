from nltk.stem import PorterStemmer
import json
import string


class InvertedIndex:
    index = {}
    docmap = {}

    def __add_document(self,doc_id,text):
        pass

    def get_documents(self,term):
        pass

    def build(self):
        pass
    
def keyword_search(search_query:str, json_path = "/home/ppappas/rag-search-engine/data/movies.json"):
    movies_dic = {}
    result = []
    with open(json_path, 'r', encoding= 'utf-8') as file:
        movies_dic = json.load(file)
    for movie in movies_dic["movies"]:
        title :str = movie["title"]
        title_lowered: str = movie["title"].lower()
        title_lowered = title_lowered.translate(str.maketrans(string.punctuation,string.punctuation,string.punctuation))
        search_query_processed = search_query.translate(str.maketrans(string.punctuation,string.punctuation,string.punctuation))
        if tokenize(search_query_processed, title_lowered) == True:
            result.append(title)
    x=1
    for title in result:
         print(f"{title}")
         x+=1

def tokenize(search_query:str, title:str):
    stemmer = PorterStemmer()
    stopwords = load_stopwords()
    search_tokens =  list(filter(lambda item: item not in stopwords, search_query.split()))
    title_tokens = list(filter(lambda item: item not in stopwords, title.split()))
    for token in search_tokens:
        if any(stemmer.stem(token).lower() in stemmer.stem(title_token).lower() for title_token in title_tokens):
             return True
    return False

def load_stopwords(file = r"/home/ppappas/rag-search-engine/data/stopwords.txt"):
    with open(file,"r") as f:
        stopwords = f.read()
        stopwords = str.splitlines(stopwords)
    tokens = []
    for word in stopwords:
        tokens.append(word
                       .translate(str.maketrans(string.punctuation,
                                                string.punctuation,
                                                string.punctuation))
                                                .lower())
    return tokens
