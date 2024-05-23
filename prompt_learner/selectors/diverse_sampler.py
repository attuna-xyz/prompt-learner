"""Picks n most diverse samples using tf-idf as a measure."""
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .selector import Selector
random.seed(42)


class DiverseSampler(Selector):
    """Picks n most diverse samples."""

    def select_examples(self):
        """Select n examples based on tf-idf diversity."""
        self.all_examples = self.task.examples
        self.selected_examples = []
        texts = [example.text for example in self.all_examples]

        # Vectorize text data
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Compute pairwise cosine similarity matrix
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Find the indices of the n most diverse samples
        indices = []
        while len(indices) < self.num_samples:
            if not indices:
                # Start with the first text if no indices are selected yet
                current_index = 0
            else:
                # Select the next text that has the minimum maximum similarity
                # to already selected texts
                current_index = np.argmin(np.max(sim_matrix[:, indices], axis=1))
            
            # Add the index of the selected text to the list
            indices.append(current_index)

        self.selected_examples = [self.all_examples[i] for i in indices]
        self.task.selected_examples = self.selected_examples
