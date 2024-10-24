import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy
import re
from typing import List, Dict, Tuple, Any
import pandas as pd

class LSIRetrieval:
    def __init__(self, n_components: int = 100, language: str = 'en_core_web_sm'):
        """
        Initialize the LSI-based retrieval system.
        
        Args:
            n_components (int): Number of latent dimensions for LSI
            language (str): spaCy language model to use
        """
        # Load spaCy model
        self.nlp = spacy.load(language)
        
        # Customize spaCy pipeline for better performance
        # Disable unnecessary components
        self.nlp.disable_pipes(['parser', 'ner'])
        
        self.n_components = n_components
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            stop_words=None,  # We'll handle stop words in tokenization
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )
        
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='randomized'
        )
        
        # Storage for processed data
        self.documents = []
        self.document_term_matrix = None
        self.lsi_matrix = None
        self.term_concept_matrix = None
        self.terms = None
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Preprocess and tokenize text using spaCy.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Preprocessed tokens
        """
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Filter tokens and lemmatize
        tokens = [
            token.lemma_
            for token in doc
            if not (token.is_stop or token.is_space or token.is_punct) and len(token.text) > 2
        ]
        
        return tokens
    
    def _preprocess_batch(self, texts: List[str], batch_size: int = 1000) -> List[str]:
        """
        Preprocess a batch of texts using spaCy's pipe for better performance.
        
        Args:
            texts (List[str]): List of texts to process
            batch_size (int): Size of batches for processing
            
        Returns:
            List[str]: Preprocessed texts
        """
        processed_texts = []
        
        # Process texts in batches
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            tokens = [
                token.lemma_
                for token in doc
                if not (token.is_stop or token.is_space or token.is_punct) and len(token.text) > 2
            ]
            processed_texts.append(' '.join(tokens))
            
        return processed_texts
    
    def _compute_term_weights(self, term_doc_matrix: np.ndarray) -> np.ndarray:
        """
        Apply term weighting schemes to the term-document matrix.
        
        Args:
            term_doc_matrix (np.ndarray): Raw term-document matrix
            
        Returns:
            np.ndarray: Weighted term-document matrix
        """
        # Log frequency weighting
        term_doc_matrix = np.log1p(term_doc_matrix)
        
        # Document length normalization
        doc_lengths = np.sqrt(np.sum(term_doc_matrix ** 2, axis=1))
        term_doc_matrix = term_doc_matrix / doc_lengths[:, np.newaxis]
        
        return term_doc_matrix
    
    def fit(self, documents: List[str], batch_size: int = 1000) -> None:
        """
        Fit the LSI model to a collection of documents.
        
        Args:
            documents (List[str]): List of document texts
            batch_size (int): Batch size for processing documents
        """
        self.documents = documents
        
        # Preprocess documents in batches
        processed_docs = self._preprocess_batch(documents, batch_size)
        
        # Create term-document matrix
        self.document_term_matrix = self.vectorizer.fit_transform(processed_docs)
        self.terms = self.vectorizer.get_feature_names_out()
        
        # Perform SVD
        self.lsi_matrix = self.svd.fit_transform(self.document_term_matrix)
        
        # Calculate term-concept matrix
        self.term_concept_matrix = self.svd.components_.T
        
        # Calculate explained variance
        explained_variance_ratio = self.svd.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(explained_variance_ratio)
        
    def transform(self, queries: List[str]) -> np.ndarray:
        """
        Transform queries into the LSI space.
        
        Args:
            queries (List[str]): List of query texts
            
        Returns:
            np.ndarray: Query vectors in LSI space
        """
        # Preprocess queries
        processed_queries = self._preprocess_batch(queries)
        
        query_term_matrix = self.vectorizer.transform(processed_queries)
        query_lsi = self.svd.transform(query_term_matrix)
        return query_lsi
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using LSI.
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Ranked list of relevant documents with metadata
        """
        # Transform query to LSI space
        query_lsi = self.transform([query])
        
        # Compute similarities using cosine similarity
        similarities = np.dot(self.lsi_matrix, query_lsi.T).flatten()
        
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Prepare results with metadata
        results = []
        for idx in top_k_indices:
            # Process the document with spaCy for additional metadata
            doc = self.nlp(self.documents[idx])
            
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx],
                'index': idx,
                'key_phrases': [chunk.text for chunk in doc.noun_chunks],
                'word_count': len(doc),
                'sentence_count': len(list(doc.sents))
            })
        
        return results
    
    def analyze_terms(self, n_top_terms: int = 10) -> pd.DataFrame:
        """
        Analyze term-concept relationships.
        
        Args:
            n_top_terms (int): Number of top terms to show per concept
            
        Returns:
            pd.DataFrame: Term-concept analysis
        """
        analysis = []
        
        for concept_idx in range(self.n_components):
            # Get term weights for this concept
            term_weights = self.term_concept_matrix[:, concept_idx]
            top_term_indices = np.argsort(np.abs(term_weights))[-n_top_terms:][::-1]
            
            for term_idx in top_term_indices:
                analysis.append({
                    'concept': concept_idx,
                    'term': self.terms[term_idx],
                    'weight': term_weights[term_idx],
                    'variance_explained': self.svd.explained_variance_ratio_[concept_idx]
                })
        
        return pd.DataFrame(analysis)
    
    def get_term_relationships(self, term: str, n_related: int = 10) -> List[Tuple[str, float]]:
        """
        Find semantically related terms based on LSI space.
        
        Args:
            term (str): Input term
            n_related (int): Number of related terms to return
            
        Returns:
            List[Tuple[str, float]]: Related terms and their similarity scores
        """
        # Lemmatize the input term using spaCy
        doc = self.nlp(term)
        if len(doc) > 0:
            term = doc[0].lemma_
        
        try:
            term_idx = np.where(self.terms == term)[0][0]
        except IndexError:
            return []
        
        term_vector = self.term_concept_matrix[term_idx]
        similarities = np.dot(self.term_concept_matrix, term_vector)
        
        top_indices = np.argsort(similarities)[-n_related-1:-1][::-1]
        return [(self.terms[idx], similarities[idx]) for idx in top_indices]

# Example usage
def demo_lsi():
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fox is a cunning animal that belongs to the canine family",
        "Dogs are loyal pets and great companions",
        "Quick reflexes are essential for survival in the wild",
        "The lazy cat sleeps all day long",
    ]
    
    # Initialize and fit LSI model
    lsi = LSIRetrieval(n_components=3)
    lsi.fit(documents)
    
    # Perform a search
    query = "fox animal"
    results = lsi.search(query, k=3)
    
    # Analyze term relationships
    term_analysis = lsi.analyze_terms(n_top_terms=5)
    
    return results, term_analysis

if __name__ == "__main__":
    # Ensure you have the spaCy model installed:
    # python -m spacy download en_core_web_sm
    
    search_results, term_analysis = demo_lsi()
    
    print("Search Results:")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['document']} (Score: {result['similarity']:.4f})")
        print(f"   Key phrases: {', '.join(result['key_phrases'])}")
        print(f"   Word count: {result['word_count']}")
        print(f"   Sentence count: {result['sentence_count']}\n")
    
    print("Term Analysis:")
    print(term_analysis)
