from search_modes import keyword_search
from keyword_search_cli import main

class InvertedIndex:
    def _init(self, index:dict, docmap:dict)
        self.index = index
        self.docmap = docmap
    

    def __add_document(self,doc_id,text):
        text = keyword_search()
        
        self.index[text]=doc_id

    def get_documents(self,term):
        pass

    def build(self):
        pass

    def save(self):
        pass