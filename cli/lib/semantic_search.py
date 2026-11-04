from sentence_transformers import SentenceTransformer

def embed_text(text):

    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def verify_model():
        search_instance = SemanticSearch()
        print(f"Model loaded: {search_instance.model}")
        print (f"Max sequence length: {search_instance.model.max_seq_length}")

    def generate_embedding(self, text):
        if not text:
            raise ValueError("No text provided")
        text = [text]
        embedded_text = self.model.encode(text)
        return embedded_text[0]
