"""
============================================================
Brain Sparks - NLP Utilities Module
============================================================
This module implements the UNDERSTAND pillar of our cognitive system.

The UNDERSTAND pillar is responsible for:
1. Parsing user queries to extract meaning
2. Identifying key topics and entities
3. Detecting local context (e.g., Uganda-specific applications)
4. Computing semantic similarity between queries and content

Author: Rugogamu Noela
Institution: Uganda Christian University (UCU)
============================================================
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import math

# ==============================================================
# NLTK-based NLP Processing
# We use NLTK as our primary NLP library because it's:
# - Lightweight and doesn't require large model downloads
# - Works offline without API calls
# - Suitable for educational demonstrations
# ==============================================================

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

# Download required NLTK data (will only download if not present)
def ensure_nltk_data():
    """
    Download required NLTK data packages.
    This function checks and downloads necessary resources for NLP processing.
    """
    required_packages = [
        'punkt',           # For tokenization
        'stopwords',       # For filtering common words
        'wordnet',         # For lemmatization (reducing words to base form)
        'averaged_perceptron_tagger',  # For part-of-speech tagging
        'punkt_tab'        # Updated punkt tokenizer
    ]
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}' if package in ['stopwords', 'wordnet'] else f'taggers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass  # Continue if download fails, we'll handle gracefully


# ==============================================================
# Topic Keywords Dictionary
# This maps educational topics to related keywords for better matching
# This is our "knowledge" about what terms relate to which topics
# ==============================================================

TOPIC_KEYWORDS = {
    'quantum_computing': [
        'quantum', 'qubit', 'superposition', 'entanglement', 'quantum computing',
        'quantum algorithm', 'quantum gate', 'quantum mechanics', 'qubits',
        'quantum computer', 'quantum physics', 'schrodinger', 'wave function'
    ],
    'machine_learning': [
        'machine learning', 'ml', 'neural network', 'deep learning', 'ai',
        'artificial intelligence', 'classification', 'regression', 'training',
        'model', 'prediction', 'supervised', 'unsupervised', 'algorithm',
        'data science', 'tensorflow', 'pytorch', 'scikit-learn'
    ],
    'data_science': [
        'data science', 'data analysis', 'analytics', 'statistics', 'pandas',
        'visualization', 'charts', 'graphs', 'data', 'insights', 'big data',
        'data mining', 'exploratory', 'eda'
    ],
    'artificial_intelligence': [
        'artificial intelligence', 'ai', 'intelligent', 'automation', 'cognitive',
        'nlp', 'natural language', 'computer vision', 'robotics', 'chatbot',
        'expert system', 'reasoning'
    ],
    'web_development': [
        'web development', 'website', 'html', 'css', 'javascript', 'frontend',
        'backend', 'api', 'rest', 'web app', 'react', 'flask', 'django',
        'responsive', 'mobile-first'
    ],
    'cybersecurity': [
        'cybersecurity', 'security', 'hacking', 'encryption', 'firewall',
        'malware', 'virus', 'penetration testing', 'vulnerability', 'password',
        'authentication', 'privacy', 'threat'
    ],
    'cloud_computing': [
        'cloud', 'aws', 'azure', 'gcp', 'serverless', 'lambda', 'iaas', 'paas',
        'saas', 'docker', 'kubernetes', 'scalability', 'cloud computing'
    ],
    'databases': [
        'database', 'sql', 'nosql', 'mongodb', 'mysql', 'postgresql', 'query',
        'schema', 'table', 'data storage', 'relational', 'crud'
    ],
    'blockchain': [
        'blockchain', 'bitcoin', 'cryptocurrency', 'smart contract', 'ethereum',
        'decentralized', 'ledger', 'mining', 'token', 'nft', 'defi'
    ],
    'internet_of_things': [
        'iot', 'internet of things', 'sensor', 'arduino', 'raspberry pi',
        'smart', 'connected', 'embedded', 'microcontroller', 'smart home'
    ],
    'digital_health': [
        'health', 'healthcare', 'medical', 'telemedicine', 'diagnosis',
        'patient', 'hospital', 'clinic', 'medicine', 'disease', 'treatment',
        'health tech', 'mhealth', 'ehr'
    ],
    'fintech': [
        'fintech', 'finance', 'banking', 'mobile money', 'payment', 'transaction',
        'lending', 'insurance', 'financial', 'money', 'wallet', 'mpesa'
    ],
    'programming': [
        'programming', 'coding', 'python', 'java', 'javascript', 'code',
        'software', 'development', 'developer', 'programmer', 'script'
    ],
    'educational_technology': [
        'education', 'learning', 'teaching', 'school', 'university', 'course',
        'student', 'teacher', 'edtech', 'lms', 'online learning', 'e-learning'
    ],
    'environmental_tech': [
        'environment', 'climate', 'sustainability', 'green', 'renewable',
        'solar', 'conservation', 'pollution', 'ecosystem', 'gis', 'satellite'
    ],
    'entrepreneurship': [
        'startup', 'entrepreneur', 'business', 'founder', 'innovation',
        'venture', 'funding', 'mvp', 'product', 'market', 'scale'
    ],
    'project_management': [
        'project', 'management', 'agile', 'scrum', 'kanban', 'team', 'sprint',
        'deadline', 'planning', 'delivery', 'roadmap'
    ],
    'design_thinking': [
        'design', 'thinking', 'ux', 'user experience', 'prototype', 'ideation',
        'empathy', 'innovation', 'creative', 'human-centered'
    ],
    'technology_ethics': [
        'ethics', 'bias', 'fairness', 'privacy', 'responsible', 'rights',
        'discrimination', 'transparency', 'accountability', 'ethical'
    ],
    'mobile_development': [
        'mobile', 'app', 'android', 'ios', 'react native', 'flutter',
        'smartphone', 'application', 'mobile app'
    ],
    'devops': [
        'devops', 'ci/cd', 'deployment', 'docker', 'kubernetes', 'pipeline',
        'automation', 'infrastructure', 'monitoring'
    ],
    'software_quality': [
        'testing', 'quality', 'qa', 'unit test', 'bug', 'debug', 'tdd',
        'test-driven', 'integration test'
    ]
}

# ==============================================================
# Uganda Context Keywords
# These keywords help identify when a query relates to Uganda
# or East African context
# ==============================================================

UGANDA_CONTEXT_KEYWORDS = {
    'location': [
        'uganda', 'ugandan', 'kampala', 'entebbe', 'jinja', 'mbarara',
        'east africa', 'african', 'africa', 'local', 'regional'
    ],
    'agriculture': [
        'farming', 'agriculture', 'crop', 'coffee', 'banana', 'matooke',
        'livestock', 'cattle', 'fish', 'farmer', 'harvest', 'soil',
        'irrigation', 'food security'
    ],
    'healthcare': [
        'malaria', 'hiv', 'aids', 'tuberculosis', 'maternal', 'child health',
        'rural health', 'clinic', 'hospital', 'doctor shortage', 'medicine'
    ],
    'finance': [
        'mobile money', 'mtn money', 'airtel money', 'unbanked', 'microfinance',
        'remittance', 'financial inclusion', 'boda boda'
    ],
    'development': [
        'development', 'poverty', 'sdg', 'sustainable', 'rural', 'village',
        'community', 'ngo', 'aid', 'empowerment'
    ],
    'education': [
        'school', 'university', 'makerere', 'ucu', 'student', 'teacher',
        'primary', 'secondary', 'literacy', 'vocational'
    ],
    'technology_access': [
        'mobile', 'smartphone', 'internet access', 'connectivity', 'data cost',
        'offline', 'low bandwidth', 'feature phone'
    ],
    'challenges': [
        'problem', 'challenge', 'issue', 'solve', 'solution', 'help',
        'improve', 'address', 'tackle', 'relevant', 'applicable'
    ]
}


class QueryParser:
    """
    The QueryParser class is the core of our UNDERSTAND pillar.
    
    It takes a user's natural language query and extracts:
    1. Main topic(s) the user wants to learn about
    2. Local context (Uganda-specific applications)
    3. Intent (explain, learn, apply, compare, etc.)
    4. Additional concepts mentioned
    
    This enables the rest of our cognitive system to:
    - Find relevant educational content
    - Generate contextually appropriate explanations
    - Recommend a personalized learning path
    """
    
    def __init__(self):
        """
        Initialize the QueryParser with NLP tools.
        """
        # Ensure we have required NLTK data
        ensure_nltk_data()
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load stopwords (common words like 'the', 'is', 'at' that don't add meaning)
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback if stopwords not available
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'it', 'be', 'as', 'by', 'this', 'that', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        
        # Add some domain-specific stop words
        self.stop_words.update(['explain', 'show', 'tell', 'want', 'learn', 'know', 'understand', 'help', 'please', 'could', 'would', 'like', 'need'])
        
    def tokenize(self, text: str) -> List[str]:
        """
        Break text into individual words (tokens).
        
        Example: "quantum computing" -> ["quantum", "computing"]
        
        Args:
            text: The input text to tokenize
            
        Returns:
            List of word tokens
        """
        try:
            # Use NLTK's word tokenizer
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.lower().split()
        
        # Remove punctuation and numbers-only tokens
        tokens = [t for t in tokens if t.isalpha()]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common words that don't contribute to meaning.
        
        Example: ["what", "is", "quantum", "computing"] -> ["quantum", "computing"]
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list without stopwords
        """
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Reduce words to their base/dictionary form.
        
        Example: ["computing", "computers"] -> ["compute", "computer"]
        
        This helps match variations of the same word.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Extract n-grams (sequences of n words) from tokens.
        
        Example with n=2: ["quantum", "computing", "basics"] 
                      -> ["quantum computing", "computing basics"]
        
        This helps match multi-word phrases like "machine learning".
        
        Args:
            tokens: List of word tokens
            n: Number of words in each n-gram
            
        Returns:
            List of n-gram strings
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return ngrams
    
    def identify_topic(self, tokens: List[str], ngrams: List[str]) -> Tuple[str, float]:
        """
        Identify the main educational topic from the query.
        
        Uses keyword matching to score each possible topic, returning
        the best match with a confidence score.
        
        Args:
            tokens: Individual word tokens from query
            ngrams: Multi-word phrases from query
            
        Returns:
            Tuple of (topic_name, confidence_score)
        """
        topic_scores = {}
        
        # Combine tokens and ngrams for matching
        all_terms = set(tokens + ngrams)
        
        # Score each topic based on keyword matches
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Check if keyword appears in our terms
                keyword_lower = keyword.lower()
                
                # Direct match
                if keyword_lower in all_terms:
                    score += 2  # Higher weight for direct match
                    matched_keywords.append(keyword)
                # Partial match (keyword contains or is contained in a term)
                else:
                    for term in all_terms:
                        if keyword_lower in term or term in keyword_lower:
                            score += 1
                            matched_keywords.append(keyword)
                            break
            
            if score > 0:
                # Normalize score by number of keywords (0-1 range)
                normalized_score = min(score / (len(keywords) * 0.5), 1.0)
                topic_scores[topic] = {
                    'score': normalized_score,
                    'matched': matched_keywords
                }
        
        # Return best matching topic
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1]['score'])
            return best_topic[0], best_topic[1]['score']
        
        return 'general', 0.0
    
    def identify_secondary_topics(self, tokens: List[str], ngrams: List[str], primary_topic: str) -> List[Tuple[str, float]]:
        """
        Identify additional related topics beyond the primary one.
        
        Args:
            tokens: Word tokens from query
            ngrams: Multi-word phrases from query
            primary_topic: Already identified main topic
            
        Returns:
            List of (topic, score) tuples for secondary topics
        """
        secondary = []
        all_terms = set(tokens + ngrams)
        
        for topic, keywords in TOPIC_KEYWORDS.items():
            if topic == primary_topic:
                continue
                
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in all_terms:
                    score += 1
                else:
                    for term in all_terms:
                        if keyword_lower in term or term in keyword_lower:
                            score += 0.5
                            break
            
            if score >= 1:  # Threshold for secondary topic
                normalized = min(score / (len(keywords) * 0.3), 1.0)
                secondary.append((topic, normalized))
        
        # Sort by score and return top 3
        secondary.sort(key=lambda x: x[1], reverse=True)
        return secondary[:3]
    
    def detect_uganda_context(self, tokens: List[str], ngrams: List[str], original_text: str) -> Dict:
        """
        Detect Uganda-specific context in the query.
        
        This is crucial for providing locally relevant recommendations.
        
        Args:
            tokens: Word tokens
            ngrams: Multi-word phrases
            original_text: Original query text
            
        Returns:
            Dictionary with Uganda context details
        """
        context = {
            'has_uganda_context': False,
            'location_mentioned': False,
            'categories': [],
            'specific_mentions': [],
            'relevance_score': 0.0
        }
        
        all_terms = set(tokens + ngrams)
        original_lower = original_text.lower()
        
        total_matches = 0
        
        for category, keywords in UGANDA_CONTEXT_KEYWORDS.items():
            category_matches = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Check in tokenized terms
                if keyword_lower in all_terms:
                    category_matches.append(keyword)
                # Check in original text (for multi-word phrases)
                elif keyword_lower in original_lower:
                    category_matches.append(keyword)
            
            if category_matches:
                context['categories'].append(category)
                context['specific_mentions'].extend(category_matches)
                total_matches += len(category_matches)
                
                if category == 'location':
                    context['location_mentioned'] = True
        
        # Calculate relevance score
        if total_matches > 0:
            context['has_uganda_context'] = True
            # More matches = higher relevance, capped at 1.0
            context['relevance_score'] = min(total_matches / 5, 1.0)
        
        # Remove duplicates from specific mentions
        context['specific_mentions'] = list(set(context['specific_mentions']))
        
        return context
    
    def detect_intent(self, text: str, tokens: List[str]) -> Dict:
        """
        Detect the user's intent from their query.
        
        Intent categories:
        - explain: User wants an explanation of a concept
        - learn: User wants to learn/study a topic
        - apply: User wants to know how to apply knowledge
        - compare: User wants to compare topics
        - solve: User has a specific problem to solve
        
        Args:
            text: Original query text
            tokens: Tokenized words
            
        Returns:
            Dictionary with intent and confidence
        """
        text_lower = text.lower()
        
        intent_patterns = {
            'explain': {
                'keywords': ['explain', 'what is', 'what are', 'define', 'meaning', 'introduction', 'basics', 'fundamental'],
                'phrases': ['tell me about', 'explain to me', 'what does', 'how does']
            },
            'learn': {
                'keywords': ['learn', 'study', 'course', 'tutorial', 'guide', 'teach', 'education'],
                'phrases': ['i want to learn', 'teach me', 'how to learn', 'where can i learn']
            },
            'apply': {
                'keywords': ['apply', 'use', 'implement', 'practical', 'real-world', 'application', 'relevant', 'useful'],
                'phrases': ['how can i use', 'how to apply', 'practical application', 'real world use']
            },
            'compare': {
                'keywords': ['compare', 'difference', 'versus', 'vs', 'better', 'which'],
                'phrases': ['what is the difference', 'compare between', 'which is better']
            },
            'solve': {
                'keywords': ['solve', 'problem', 'issue', 'fix', 'solution', 'help', 'challenge'],
                'phrases': ['how to solve', 'help me with', 'i need to fix']
            }
        }
        
        intent_scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 1
                elif keyword in tokens:
                    score += 0.5
            
            # Check phrases (higher weight)
            for phrase in patterns['phrases']:
                if phrase in text_lower:
                    score += 2
            
            if score > 0:
                intent_scores[intent] = score
        
        # Default to 'learn' if no clear intent
        if not intent_scores:
            return {'primary': 'learn', 'confidence': 0.5, 'secondary': []}
        
        # Get primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0]
        
        # Normalize confidence
        max_possible = 5  # Rough max score
        confidence = min(primary_intent[1] / max_possible, 1.0)
        
        secondary = [i[0] for i in sorted_intents[1:3] if i[1] > 0.5]
        
        return {
            'primary': primary_intent[0],
            'confidence': confidence,
            'secondary': secondary
        }
    
    def extract_additional_concepts(self, tokens: List[str], primary_topic: str) -> List[str]:
        """
        Extract additional concepts or modifiers from the query.
        
        These help refine recommendations (e.g., "beginner", "advanced", "practical").
        
        Args:
            tokens: Word tokens
            primary_topic: Main identified topic
            
        Returns:
            List of additional concept strings
        """
        # Concept categories
        difficulty_terms = ['beginner', 'basic', 'intermediate', 'advanced', 'expert', 'introduction', 'intro', 'fundamental']
        format_terms = ['video', 'article', 'tutorial', 'course', 'book', 'quiz', 'practical', 'hands-on', 'exercise']
        scope_terms = ['overview', 'deep', 'comprehensive', 'quick', 'detailed', 'summary']
        
        concepts = []
        
        for token in tokens:
            if token in difficulty_terms:
                concepts.append(f"difficulty:{token}")
            elif token in format_terms:
                concepts.append(f"format:{token}")
            elif token in scope_terms:
                concepts.append(f"scope:{token}")
        
        return concepts
    
    def parse_query(self, query: str) -> Dict:
        """
        Main method to parse a user query and extract all relevant information.
        
        This is the primary entry point for the UNDERSTAND pillar.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Dictionary containing:
            - original_query: The input query
            - tokens: Tokenized words
            - clean_tokens: Tokens after stopword removal and lemmatization
            - primary_topic: Main educational topic
            - topic_confidence: Confidence in topic identification
            - secondary_topics: Other related topics
            - uganda_context: Uganda-specific context information
            - intent: User's intent (explain, learn, apply, etc.)
            - additional_concepts: Difficulty, format preferences
        """
        # Step 1: Tokenize
        tokens = self.tokenize(query)
        
        # Step 2: Create bigrams and trigrams for phrase matching
        bigrams = self.extract_ngrams(tokens, 2)
        trigrams = self.extract_ngrams(tokens, 3)
        all_ngrams = bigrams + trigrams
        
        # Step 3: Clean tokens (remove stopwords, lemmatize)
        clean_tokens = self.remove_stopwords(tokens)
        clean_tokens = self.lemmatize(clean_tokens)
        
        # Step 4: Identify primary topic
        primary_topic, topic_confidence = self.identify_topic(clean_tokens, all_ngrams)
        
        # Step 5: Identify secondary topics
        secondary_topics = self.identify_secondary_topics(clean_tokens, all_ngrams, primary_topic)
        
        # Step 6: Detect Uganda context
        uganda_context = self.detect_uganda_context(tokens, all_ngrams, query)
        
        # Step 7: Detect intent
        intent = self.detect_intent(query, tokens)
        
        # Step 8: Extract additional concepts
        additional_concepts = self.extract_additional_concepts(tokens, primary_topic)
        
        return {
            'original_query': query,
            'tokens': tokens,
            'clean_tokens': clean_tokens,
            'primary_topic': primary_topic,
            'topic_confidence': topic_confidence,
            'secondary_topics': secondary_topics,
            'uganda_context': uganda_context,
            'intent': intent,
            'additional_concepts': additional_concepts
        }


class SemanticSimilarity:
    """
    Computes semantic similarity between texts using TF-IDF.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) is a technique that:
    - Measures how important a word is to a document
    - Words that appear frequently in a document but rarely in others get higher scores
    - This helps match queries to relevant content
    """
    
    def __init__(self):
        """Initialize with empty vocabulary."""
        self.vocabulary = {}
        self.idf = {}
        self.documents = []
        
    def fit(self, documents: List[str]):
        """
        Build vocabulary and compute IDF from a set of documents.
        
        Args:
            documents: List of text documents
        """
        self.documents = documents
        word_doc_count = Counter()
        
        # Build vocabulary and count document frequencies
        for doc in documents:
            # Tokenize and get unique words in this document
            try:
                words = set(word_tokenize(doc.lower()))
            except:
                words = set(doc.lower().split())
            
            for word in words:
                if word.isalpha():  # Only alphabetic words
                    word_doc_count[word] += 1
                    if word not in self.vocabulary:
                        self.vocabulary[word] = len(self.vocabulary)
        
        # Compute IDF (Inverse Document Frequency)
        # IDF = log(total_documents / documents_containing_word)
        # Higher IDF = more rare/important word
        n_docs = len(documents)
        for word, count in word_doc_count.items():
            self.idf[word] = math.log(n_docs / (count + 1)) + 1
    
    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """
        Compute TF-IDF vector for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping words to TF-IDF scores
        """
        try:
            words = word_tokenize(text.lower())
        except:
            words = text.lower().split()
        
        # Compute term frequencies
        word_count = Counter(w for w in words if w.isalpha())
        total_words = sum(word_count.values())
        
        tfidf = {}
        for word, count in word_count.items():
            tf = count / total_words if total_words > 0 else 0
            idf = self.idf.get(word, 1.0)
            tfidf[word] = tf * idf
        
        return tfidf
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.
        
        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (very similar)
        - 0.0 = perpendicular (no similarity)
        
        Args:
            vec1: First TF-IDF vector
            vec2: Second TF-IDF vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Find common words
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[w] * vec2[w] for w in common_words)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        vec1 = self.compute_tfidf(text1)
        vec2 = self.compute_tfidf(text2)
        return self.cosine_similarity(vec1, vec2)


def generate_topic_explanation(topic: str, uganda_context: Dict) -> str:
    """
    Generate an explanation of a topic with optional Uganda context.
    
    This function creates a structured explanation based on the identified topic.
    In a production system, this might use a language model or knowledge base.
    
    Args:
        topic: The educational topic
        uganda_context: Uganda context dictionary from query parsing
        
    Returns:
        A string explanation of the topic
    """
    # Topic explanations (simplified knowledge base)
    explanations = {
        'quantum_computing': """
Quantum Computing is a revolutionary approach to computation that harnesses the principles of quantum mechanics—the physics of very small things like atoms and electrons.

**Key Concepts:**
1. **Qubits**: Unlike classical bits (0 or 1), qubits can be in a "superposition" of both states simultaneously
2. **Superposition**: A qubit can represent multiple values at once, enabling parallel processing
3. **Entanglement**: Quantum particles can be connected so that measuring one instantly affects the other
4. **Quantum Gates**: Operations that manipulate qubits, similar to logic gates in classical computing

**Why It Matters:**
Quantum computers can solve certain problems exponentially faster than classical computers, including:
- Breaking current encryption (and creating unbreakable new encryption)
- Simulating molecules for drug discovery
- Optimizing complex logistics and supply chains
- Machine learning and pattern recognition
        """,
        
        'machine_learning': """
Machine Learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed.

**Key Concepts:**
1. **Supervised Learning**: Learning from labeled examples (e.g., classifying images as "cat" or "dog")
2. **Unsupervised Learning**: Finding patterns in unlabeled data (e.g., customer segmentation)
3. **Neural Networks**: Computing systems inspired by biological brains
4. **Training**: The process of teaching a model using data

**Common Applications:**
- Recommendation systems (Netflix, Spotify)
- Spam detection in email
- Image and speech recognition
- Predictive analytics for business
        """,
        
        'data_science': """
Data Science is the interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.

**Key Components:**
1. **Data Collection**: Gathering data from various sources
2. **Data Cleaning**: Preparing data for analysis by handling missing values and errors
3. **Exploratory Analysis**: Understanding patterns and distributions in data
4. **Visualization**: Creating charts and graphs to communicate insights

**Tools & Technologies:**
- Python (pandas, numpy, matplotlib)
- SQL for database queries
- Statistical analysis methods
- Business intelligence dashboards
        """,
        
        'artificial_intelligence': """
Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence.

**Major Areas:**
1. **Natural Language Processing (NLP)**: Understanding and generating human language
2. **Computer Vision**: Interpreting images and video
3. **Robotics**: Physical machines that can sense and act
4. **Expert Systems**: Programs that mimic human expertise in specific domains

**AI Approaches:**
- Rule-based systems (explicit programming)
- Machine learning (learning from data)
- Deep learning (neural networks with many layers)
        """,
        
        'web_development': """
Web Development is the process of building and maintaining websites and web applications.

**Core Technologies:**
1. **HTML**: Structure and content of web pages
2. **CSS**: Styling and visual presentation
3. **JavaScript**: Interactivity and dynamic behavior
4. **Backend Languages**: Python, Node.js, PHP for server-side logic

**Development Types:**
- Frontend: User-facing interface
- Backend: Server, database, and application logic
- Full-stack: Both frontend and backend
        """,
        
        'cybersecurity': """
Cybersecurity is the practice of protecting computer systems, networks, and data from digital attacks and unauthorized access.

**Key Areas:**
1. **Network Security**: Protecting computer networks from intrusions
2. **Application Security**: Making software secure from threats
3. **Information Security**: Protecting data integrity and privacy
4. **Operational Security**: Processes for handling and protecting data

**Common Threats:**
- Malware (viruses, ransomware)
- Phishing attacks
- Data breaches
- Social engineering
        """,
        
        'cloud_computing': """
Cloud Computing delivers computing services—servers, storage, databases, networking, software—over the internet ("the cloud").

**Service Models:**
1. **IaaS** (Infrastructure as a Service): Rent virtual machines and storage
2. **PaaS** (Platform as a Service): Platforms for developing and deploying apps
3. **SaaS** (Software as a Service): Ready-to-use applications online

**Benefits:**
- No upfront infrastructure costs
- Scalability on demand
- Global accessibility
- Automatic updates and maintenance
        """,
        
        'blockchain': """
Blockchain is a decentralized, distributed digital ledger that records transactions across multiple computers.

**Key Features:**
1. **Decentralization**: No single authority controls the network
2. **Immutability**: Once recorded, data cannot be altered
3. **Transparency**: All transactions are visible to participants
4. **Smart Contracts**: Self-executing contracts with coded terms

**Applications Beyond Cryptocurrency:**
- Supply chain tracking
- Digital identity management
- Healthcare records
- Land registry systems
        """,
        
        'internet_of_things': """
Internet of Things (IoT) refers to the network of physical devices embedded with sensors, software, and connectivity to exchange data.

**Components:**
1. **Sensors**: Collect data from the environment (temperature, motion, etc.)
2. **Connectivity**: Wi-Fi, Bluetooth, cellular networks
3. **Data Processing**: Edge computing or cloud analysis
4. **User Interface**: Dashboards and mobile apps

**Applications:**
- Smart homes and cities
- Industrial automation
- Agricultural monitoring
- Healthcare devices
        """,
        
        'digital_health': """
Digital Health encompasses the use of digital technologies to improve health and healthcare delivery.

**Key Areas:**
1. **Telemedicine**: Remote medical consultations
2. **mHealth**: Health apps and mobile health solutions
3. **Electronic Health Records**: Digital patient information systems
4. **AI Diagnostics**: Machine learning for disease detection

**Benefits:**
- Improved access to healthcare
- Better patient outcomes
- Reduced costs
- Data-driven decision making
        """,
        
        'fintech': """
Financial Technology (FinTech) refers to technology that improves and automates financial services.

**Key Innovations:**
1. **Mobile Payments**: Pay using smartphones
2. **Digital Banking**: Banking services without physical branches
3. **P2P Lending**: Direct lending between individuals
4. **Insurtech**: Technology in insurance

**Impact:**
- Financial inclusion for the unbanked
- Lower transaction costs
- Faster, more convenient services
- New business models
        """,
        
        'general': """
Educational Technology encompasses various fields that apply technology to solve problems and improve lives.

The field you're interested in involves using computational methods and digital tools to address real-world challenges. Modern technology education covers programming, data analysis, artificial intelligence, and their applications across industries like healthcare, agriculture, finance, and education.
        """
    }
    
    # Get base explanation
    explanation = explanations.get(topic, explanations['general'])
    
    return explanation.strip()


def generate_uganda_relevance(topic: str, uganda_context: Dict) -> str:
    """
    Generate Uganda-specific relevance and applications for a topic.
    
    Args:
        topic: The educational topic
        uganda_context: Uganda context from query parsing
        
    Returns:
        String describing Uganda relevance
    """
    # Uganda-specific applications by topic
    uganda_applications = {
        'quantum_computing': """
Relevance to Uganda:

While quantum computing is still emerging, it holds significant potential for Uganda:

 **Agriculture Optimization**
- Quantum algorithms could optimize crop distribution across Uganda's diverse agricultural regions
- Better modeling of soil conditions and weather patterns for the 70% of Ugandans who depend on farming

 **Healthcare Research**
- Accelerate drug discovery for tropical diseases prevalent in Uganda (malaria, HIV/AIDS)
- Quantum simulations could model protein interactions for vaccine development

 **E-Government Security**
- Future-proof Uganda's national ID system and digital government services
- Quantum-resistant cryptography for secure mobile money transactions

 **Climate Modeling**
- Better predictions for Lake Victoria water levels and regional climate patterns
- Optimize renewable energy distribution across the national grid

 **Telecommunications**
- Quantum communication could eventually enable ultra-secure communications
- Optimize network routing for improved connectivity in rural areas
        """,
        
        'machine_learning': """
**Relevance to Uganda:**

Machine learning has immediate, practical applications for Uganda's development:

 **Smart Agriculture**
- Predict crop yields for coffee, bananas, and maize based on weather and soil data
- Detect crop diseases from smartphone photos (banana bacterial wilt, cassava mosaic)
- Optimize irrigation schedules for water conservation

 **Healthcare Solutions**
- Diagnose malaria from blood smear images in rural clinics with limited doctors
- Predict disease outbreaks from health facility data
- Personalize HIV treatment recommendations

 **Financial Inclusion**
- Credit scoring for the unbanked using mobile money transaction patterns
- Fraud detection for Uganda's growing mobile money ecosystem
- Personalized financial advice for small businesses

 **Education Enhancement**
- Personalized learning paths for students in under-resourced schools
- Automatic grading and feedback for teachers
- Predict student dropout risk for early intervention
        """,
        
        'data_science': """
Relevance to Uganda:

Data science skills are increasingly valuable for Uganda's development:

 **Evidence-Based Policy**
- Analyze census and survey data to inform government decisions
- Track progress on Sustainable Development Goals (SDGs)
- Monitor public health indicators and disease trends

 **Business Intelligence**
- Help Ugandan SMEs understand customer behavior
- Analyze market trends for agriculture exports
- Optimize supply chains for local manufacturers

 **Development Impact**
- NGOs can measure program effectiveness with data
- Track poverty reduction and education outcomes
- Improve targeting of social protection programs

 **Digital Economy**
- Analyze mobile money patterns to understand economic activity
- Study internet usage to improve digital services
- Support Uganda's growing tech startup ecosystem
        """,
        
        'artificial_intelligence': """
Relevance to Uganda:

AI can address key challenges facing Uganda:

 **Local Language Technology**
- Develop AI systems that understand Luganda, Runyankole, and other local languages
- Voice interfaces for citizens with limited literacy
- Preserve and digitize Uganda's linguistic heritage

 **Healthcare Access**
- AI chatbots for health information in local languages
- Automated triage to help overburdened health facilities
- Remote diagnostics for rural communities

 **Agricultural Advisory**
- AI-powered advice for farmers via SMS or voice
- Pest and disease identification from photos
- Market price predictions for better selling decisions

 **Government Services**
- Automated responses for citizen inquiries
- Document processing and verification
- Smart traffic management for Kampala
        """,
        
        'web_development': """
**Relevance to Uganda:**

Web development skills directly support Uganda's digital economy:

 **E-Commerce**
- Build online stores for Ugandan businesses to reach wider markets
- Create platforms connecting farmers directly to buyers
- Enable artisans and craftspeople to sell internationally

 **Mobile-First Design**
- 27+ million mobile subscribers make mobile-first design essential
- Progressive web apps that work on low-cost smartphones
- Offline-capable apps for areas with limited connectivity

 **Business Digitization**
- Help traditional businesses go digital
- Create management systems for schools, clinics, and businesses
- Build booking and appointment systems

 **Employment & Freelancing**
- Web development skills enable remote work opportunities
- Access to global freelancing platforms
- Growing demand from Uganda's tech startup scene
        """,
        
        'cybersecurity': """
Relevance to Uganda:

Cybersecurity is critical as Uganda digitizes:

 **Mobile Money Protection**
- Secure the 27+ million mobile money users from fraud
- Protect against SIM swap attacks and phishing
- Ensure safe digital transactions

 **Government Systems**
- Protect the National ID database
- Secure e-government services
- Defend against cyber attacks on critical infrastructure

 **Business Security**
- Help Ugandan businesses protect customer data
- Implement secure payment systems
- Train staff on security awareness

 **Personal Privacy**
- Protect citizens' digital rights and privacy
- Secure communication for journalists and activists
- Educate about safe online practices
        """,
        
        'internet_of_things': """
Relevance to Uganda:

IoT has transformative potential for Uganda:

 **Smart Farming**
- Soil moisture sensors for efficient irrigation
- Weather stations for local farming communities
- Livestock tracking and health monitoring

 **Energy Management**
- Smart solar systems for rural electrification
- Monitoring of off-grid energy installations
- Efficient power distribution

 **Water & Sanitation**
- Monitor water quality in real-time
- Track water levels in tanks and boreholes
- Smart waste management systems

 **Healthcare Monitoring**
- Remote patient monitoring devices
- Temperature tracking for vaccine cold chains
- Air quality monitoring in urban areas
        """,
        
        'fintech': """
Relevance to Uganda:

Uganda is a fintech success story with huge potential:

 **Mobile Money Innovation**
- Uganda has 27+ million registered mobile money users
- Opportunity to build on this infrastructure
- Create new services for existing mobile money users

 **Financial Inclusion**
- Reach the 78% of Ugandans still underbanked
- Microfinance solutions for smallholder farmers
- Savings products designed for irregular incomes

 **Remittances**
- Cheaper, faster international transfers
- Diaspora connections to family in Uganda
- Cross-border payments in East Africa

 **Alternative Lending**
- Credit for SMEs without traditional collateral
- Agricultural loans tied to crop cycles
- Peer-to-peer lending platforms
        """,
        
        'general': """
Relevance to Uganda:

Technology education is crucial for Uganda's future:

 **Economic Growth**
- Tech skills enable participation in the global digital economy
- Support Uganda's vision for middle-income status
- Create employment for the young population

 **Development Goals**
- Technology can accelerate progress on all 17 SDGs
- Address challenges in health, education, and agriculture
- Build resilient systems for the future

 **Innovation Ecosystem**
- Growing tech startup scene in Kampala
- Innovation hubs and incubators
- Partnership opportunities with global tech companies
        """
    }
    
    relevance = uganda_applications.get(topic, uganda_applications['general'])
    return relevance.strip()


# ==============================================================
# Utility Functions for External Use
# ==============================================================

def preprocess_text(text: str) -> str:
    """
    Simple text preprocessing: lowercase and remove extra whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    return ' '.join(text.lower().split())


def extract_keywords(text: str, n: int = 10) -> List[str]:
    """
    Extract top keywords from text.
    
    Args:
        text: Input text
        n: Number of keywords to return
        
    Returns:
        List of keywords
    """
    ensure_nltk_data()
    
    try:
        tokens = word_tokenize(text.lower())
    except:
        tokens = text.lower().split()
    
    # Remove stopwords and non-alphabetic tokens
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    keywords = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    
    # Return most common
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(n)]


# ==============================================================
# Main execution for testing
# ==============================================================

if __name__ == "__main__":
    # Test the query parser
    parser = QueryParser()
    
    test_queries = [
        "Explain the basics of quantum computing and show me how it could be relevant for solving problems in Uganda.",
        "I want to learn machine learning for agriculture",
        "How can cybersecurity help protect mobile money in Uganda?",
        "What is artificial intelligence?"
    ]
    
    print("=" * 60)
    print("Brain Sparks - NLP Query Parser Test")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n Query: {query}")
        print("-" * 40)
        
        result = parser.parse_query(query)
        
        print(f" Primary Topic: {result['primary_topic']} (confidence: {result['topic_confidence']:.2f})")
        print(f" Secondary Topics: {result['secondary_topics']}")
        print(f" Uganda Context: {result['uganda_context']['has_uganda_context']}")
        if result['uganda_context']['has_uganda_context']:
            print(f"   Categories: {result['uganda_context']['categories']}")
            print(f"   Mentions: {result['uganda_context']['specific_mentions']}")
        print(f" Intent: {result['intent']['primary']} (confidence: {result['intent']['confidence']:.2f})")
        print(f" Additional: {result['additional_concepts']}")


