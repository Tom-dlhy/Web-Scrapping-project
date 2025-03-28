import re
import spacy
import pandas as pd

class TextProcessor:
    
    def __init__(self, dataset: pd.DataFrame, text_column: str):
        """
        Initialise la classe avec un DataFrame et la colonne contenant le texte.
        """
        self.nlp = spacy.load("en_core_web_sm")  # Chargement du modèle NLP
        self.dataset = dataset
        self.text_column = text_column
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoie le texte : suppression des caractères spéciaux et conversion en minuscules.
        """
        text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)
 
        return text.lower().strip()

    def remove_stopwords_and_lemmatize(self, token_list: list) -> list:
        """
        Retire les stopwords et applique la lemmatisation sur une liste de tokens.
        """
        stop_words = self.nlp.Defaults.stop_words
        pattern = re.compile(r'.*[A-Za-z0-9].*')
        
        # Filtrer les tokens et les convertir en minuscules
        cleaned_tokens = [
            token.lower() for token in token_list 
            if token.lower() not in stop_words and pattern.match(token)
        ]
        
        # Tokenisation avec spaCy pour la lemmatisation
        doc = self.nlp(" ".join(cleaned_tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]
        
        return lemmatized_tokens

    def preprocess_text_column(self) -> pd.DataFrame:
        """
        Applique le nettoyage, la tokenisation et la lemmatisation sur la colonne textuelle du DataFrame.
        """
        self.dataset[self.text_column] = self.dataset[self.text_column].astype(str)  # Convertir en string
        
        # Nettoyage de texte
        self.dataset["clean_text"] = self.dataset[self.text_column].apply(self.clean_text)

        # Tokenisation
        self.dataset["tokens"] = self.dataset["clean_text"].apply(lambda x: x.split())

        # Stopwords et Lemmatisation
        self.dataset["lemmatized_tokens"] = self.dataset["tokens"].apply(self.remove_stopwords_and_lemmatize)

        return self.dataset