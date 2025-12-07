"""
============================================================
Brain Sparks - Recommendation Engine Module
============================================================
This module implements the LEARN pillar of our cognitive system, plus
the core recommendation logic that ties together UNDERSTAND and REASON.

The LEARN pillar is responsible for:
1. Collecting user feedback on recommendations
2. Storing feedback persistently
3. Using feedback to improve future recommendations
4. Implementing basic personalization

The Recommendation Engine:
1. Uses content-based filtering (match query to content descriptions)
2. Incorporates knowledge graph relationships
3. Adjusts scores based on user feedback history
4. Creates personalized learning paths

Author: Rugogamu Noela
Institution: Uganda Christian University (UCU)
============================================================
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import our other modules
from nlp_utils import QueryParser, SemanticSimilarity, preprocess_text, extract_keywords
from nlp_utils import generate_topic_explanation, generate_uganda_relevance
from kg_utils import EducationalKnowledgeGraph, build_knowledge_graph_from_data, reason_about_query

# ==============================================================
# Content-Based Recommender
# ==============================================================

class ContentBasedRecommender:
    """
    A content-based recommendation system for educational resources.
    
    Content-based filtering works by:
    1. Analyzing the characteristics of items (educational resources)
    2. Building a profile of user preferences
    3. Recommending items similar to what the user liked
    
    In our case:
    - We match the user's query to resource descriptions
    - We use TF-IDF to compute similarity scores
    - We incorporate knowledge graph relationships
    - We adjust based on past feedback
    """
    
    def __init__(self, resources: List[Dict]):
        """
        Initialize the recommender with educational resources.
        
        Args:
            resources: List of resource dictionaries from our dataset
        """
        self.resources = {r['id']: r for r in resources}
        self.resource_list = resources
        
        # Build TF-IDF model from resource descriptions
        self.similarity_model = SemanticSimilarity()
        
        # Create document for each resource (combine title, description, tags)
        self.resource_documents = {}
        documents = []
        
        for resource in resources:
            # Combine relevant text fields for matching
            doc_text = ' '.join([
                resource.get('title', ''),
                resource.get('description', ''),
                ' '.join(resource.get('tags', [])),
                ' '.join(resource.get('subtopics', [])),
                resource.get('topic', ''),
                resource.get('uganda_applications', '')
            ])
            self.resource_documents[resource['id']] = doc_text
            documents.append(doc_text)
        
        # Fit the TF-IDF model on all documents
        self.similarity_model.fit(documents)
        
        # Pre-compute TF-IDF vectors for all resources
        self.resource_vectors = {
            rid: self.similarity_model.compute_tfidf(doc)
            for rid, doc in self.resource_documents.items()
        }
    
    def compute_query_similarity(self, query: str) -> Dict[str, float]:
        """
        Compute similarity between query and all resources.
        
        Args:
            query: User's search query
            
        Returns:
            Dictionary mapping resource IDs to similarity scores
        """
        # Get TF-IDF vector for query
        query_vector = self.similarity_model.compute_tfidf(query)
        
        # Compute similarity with each resource
        similarities = {}
        for rid, resource_vector in self.resource_vectors.items():
            sim = self.similarity_model.cosine_similarity(query_vector, resource_vector)
            similarities[rid] = sim
        
        return similarities
    
    def get_recommendations(self, query: str, top_k: int = 10,
                           topic_filter: str = None,
                           difficulty_filter: str = None,
                           type_filter: str = None) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations based on query similarity.
        
        Args:
            query: User's search query
            top_k: Number of recommendations to return
            topic_filter: Optional filter by topic
            difficulty_filter: Optional filter by difficulty
            type_filter: Optional filter by resource type
            
        Returns:
            List of (resource_id, score) tuples, sorted by score
        """
        # Compute similarities
        similarities = self.compute_query_similarity(query)
        
        # Apply filters
        filtered_resources = []
        for rid, score in similarities.items():
            resource = self.resources.get(rid)
            if not resource:
                continue
                
            # Apply topic filter
            if topic_filter and resource.get('topic') != topic_filter:
                continue
            
            # Apply difficulty filter
            if difficulty_filter and resource.get('difficulty') != difficulty_filter:
                continue
            
            # Apply type filter
            if type_filter and resource.get('type') != type_filter:
                continue
            
            filtered_resources.append((rid, score))
        
        # Sort by score descending
        filtered_resources.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_resources[:top_k]


# ==============================================================
# Feedback Manager (LEARN Pillar)
# ==============================================================

class FeedbackManager:
    """
    Manages user feedback for the recommendation system.
    
    The LEARN pillar works by:
    1. Collecting ratings when users interact with recommendations
    2. Storing feedback persistently in a JSON file
    3. Using feedback to adjust recommendation scores
    4. Tracking which resources are most helpful
    
    Privacy Note: We only store anonymous feedback (no user identification).
    """
    
    def __init__(self, feedback_path: str):
        """
        Initialize the feedback manager.
        
        Args:
            feedback_path: Path to feedback.json file
        """
        self.feedback_path = feedback_path
        self.feedback_data = self._load_feedback()
        
        # Compute aggregated scores for quick access
        self._compute_aggregates()
    
    def _load_feedback(self) -> Dict:
        """Load feedback from JSON file."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # Return default structure if file doesn't exist
        return {
            'metadata': {
                'description': 'User feedback storage for Brain Sparks',
                'version': '1.0'
            },
            'feedback_entries': []
        }
    
    def _save_feedback(self):
        """Save feedback to JSON file."""
        try:
            with open(self.feedback_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save feedback: {e}")
    
    def _compute_aggregates(self):
        """Compute aggregated ratings for each resource."""
        self.resource_ratings = defaultdict(list)
        self.resource_avg_rating = {}
        self.resource_helpful_count = defaultdict(int)
        
        for entry in self.feedback_data.get('feedback_entries', []):
            rid = entry.get('resource_id')
            if rid:
                # Track ratings (1-5 scale)
                if 'rating' in entry:
                    self.resource_ratings[rid].append(entry['rating'])
                
                # Track helpfulness
                if entry.get('helpful'):
                    self.resource_helpful_count[rid] += 1
        
        # Compute averages
        for rid, ratings in self.resource_ratings.items():
            self.resource_avg_rating[rid] = sum(ratings) / len(ratings)
    
    def add_feedback(self, resource_id: str, rating: int = None,
                     helpful: bool = None, comment: str = None,
                     query: str = None):
        """
        Add user feedback for a resource.
        
        Args:
            resource_id: ID of the resource being rated
            rating: 1-5 star rating
            helpful: Was this recommendation helpful?
            comment: Optional text feedback
            query: Original query that led to this recommendation
        """
        entry = {
            'resource_id': resource_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        if rating is not None:
            entry['rating'] = max(1, min(5, rating))  # Clamp to 1-5
        
        if helpful is not None:
            entry['helpful'] = helpful
        
        if comment:
            entry['comment'] = comment[:500]  # Limit comment length
        
        if query:
            entry['query'] = query[:200]  # Store query for learning
        
        self.feedback_data['feedback_entries'].append(entry)
        
        # Save and recompute aggregates
        self._save_feedback()
        self._compute_aggregates()
    
    def get_resource_score_adjustment(self, resource_id: str) -> float:
        """
        Get a score adjustment based on feedback for a resource.
        
        Resources with good ratings get a boost, bad ratings get a penalty.
        
        Args:
            resource_id: ID of resource
            
        Returns:
            Score adjustment factor (0.5 to 1.5)
        """
        if resource_id not in self.resource_avg_rating:
            return 1.0  # No adjustment for resources without feedback
        
        avg_rating = self.resource_avg_rating[resource_id]
        
        # Convert 1-5 scale to 0.5-1.5 multiplier
        # Rating 3 = 1.0 (no change)
        # Rating 5 = 1.5 (50% boost)
        # Rating 1 = 0.5 (50% penalty)
        adjustment = 0.5 + (avg_rating - 1) * 0.25
        
        return adjustment
    
    def get_total_feedback_count(self) -> int:
        """Get total number of feedback entries."""
        return len(self.feedback_data.get('feedback_entries', []))
    
    def get_resource_feedback_summary(self, resource_id: str) -> Dict:
        """
        Get feedback summary for a specific resource.
        
        Args:
            resource_id: ID of resource
            
        Returns:
            Dictionary with feedback statistics
        """
        ratings = self.resource_ratings.get(resource_id, [])
        
        return {
            'total_ratings': len(ratings),
            'average_rating': self.resource_avg_rating.get(resource_id, 0),
            'helpful_count': self.resource_helpful_count.get(resource_id, 0)
        }


# ==============================================================
# Main Cognitive Recommender System
# ==============================================================

class CognitiveRecommender:
    """
    The main cognitive recommendation system that integrates all pillars:
    
    - UNDERSTAND: Uses QueryParser to understand user queries
    - REASON: Uses KnowledgeGraph to reason about content relationships
    - LEARN: Uses FeedbackManager to improve over time
    - INTERACT: Provides a clean interface for the Streamlit app
    
    This class is the brain of Brain Sparks!
    """
    
    def __init__(self, data_path: str, feedback_path: str):
        """
        Initialize the cognitive recommender with all components.
        
        Args:
            data_path: Path to educational_content.json
            feedback_path: Path to feedback.json
        """
        # Load educational content
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.resources = self.data.get('resources', [])
        
        # Initialize the four pillars
        print("Initializing Brain Sparks Cognitive System...")
        
        # UNDERSTAND: NLP Query Parser
        print("    Loading NLP components (Understand Pillar)...")
        self.query_parser = QueryParser()
        
        # REASON: Knowledge Graph
        print("    Building Knowledge Graph (Reason Pillar)...")
        self.knowledge_graph = build_knowledge_graph_from_data(data_path)
        
        # REASON: Content-Based Recommender
        print("    Initializing Recommender Engine...")
        self.content_recommender = ContentBasedRecommender(self.resources)
        
        # LEARN: Feedback Manager
        print("    Loading Feedback System (Learn Pillar)...")
        self.feedback_manager = FeedbackManager(feedback_path)
        
        print(" Brain Sparks is ready!")
    
    def process_query(self, query: str) -> Dict:
        """
        Process a user query and generate recommendations.
        
        This is the main entry point that runs the full cognitive cycle:
        1. UNDERSTAND: Parse the query to extract meaning
        2. REASON: Use knowledge graph and content matching
        3. LEARN: Adjust scores based on feedback
        4. Return structured recommendations
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with complete response including:
            - parsed_query: NLP analysis results
            - topic_explanation: Generated explanation
            - uganda_relevance: Uganda-specific applications
            - learning_path: Recommended sequence of resources
            - reasoning_trace: How the system arrived at recommendations
        """
        response = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        try:
            # ===== UNDERSTAND PILLAR =====
            # Parse the query to understand what the user wants
            parsed_query = self.query_parser.parse_query(query)
            response['parsed_query'] = {
                'primary_topic': parsed_query['primary_topic'],
                'topic_confidence': parsed_query['topic_confidence'],
                'secondary_topics': parsed_query['secondary_topics'],
                'intent': parsed_query['intent'],
                'uganda_context': parsed_query['uganda_context'],
                'additional_concepts': parsed_query['additional_concepts']
            }
            
            # ===== REASON PILLAR =====
            # Use knowledge graph for reasoning
            kg_reasoning = reason_about_query(self.knowledge_graph, parsed_query)
            response['reasoning'] = {
                'topic_found': kg_reasoning['topic_in_graph'],
                'related_topics': kg_reasoning['related_topics'][:5],
                'reasoning_path': kg_reasoning['reasoning_path']
            }
            
            # Get content-based recommendations
            topic = parsed_query['primary_topic']
            
            # Compute query similarity scores
            similarity_scores = self.content_recommender.compute_query_similarity(query)
            
            # Get resources for the topic from knowledge graph
            topic_resources = self.knowledge_graph.get_resources_for_topic(topic)
            
            # Combine scores: topic match + similarity + feedback adjustment
            combined_scores = {}
            
            for resource in self.resources:
                rid = resource['id']
                score = 0.0
                
                # Base similarity score (0-1)
                sim_score = similarity_scores.get(rid, 0)
                score += sim_score * 0.4  # 40% weight
                
                # Topic match bonus
                if resource.get('topic') == topic:
                    score += 0.3  # 30% bonus for exact topic match
                
                # Uganda relevance bonus
                if parsed_query['uganda_context']['has_uganda_context']:
                    uganda_relevance = resource.get('uganda_relevance', [])
                    uganda_apps = resource.get('uganda_applications', '')
                    
                    for category in parsed_query['uganda_context']['categories']:
                        if category in uganda_relevance or category in uganda_apps.lower():
                            score += 0.15
                            break
                
                # Subtopic match bonus
                for subtopic in resource.get('subtopics', []):
                    if subtopic in [t[0] for t in parsed_query['secondary_topics']]:
                        score += 0.1
                        break
                
                # Apply feedback adjustment (LEARN pillar)
                feedback_adjustment = self.feedback_manager.get_resource_score_adjustment(rid)
                score *= feedback_adjustment
                
                combined_scores[rid] = score
            
            # Sort and get top recommendations
            sorted_resources = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # ===== BUILD LEARNING PATH =====
            # Create a structured learning path with:
            # - Beginner resource (Start Here)
            # - Intermediate resource (Build Understanding)
            # - Quiz (Test Yourself)
            
            learning_path = []
            selected_types = {'article': False, 'video': False, 'quiz': False}
            
            # Group by type and difficulty
            beginner_resources = []
            intermediate_resources = []
            advanced_resources = []
            quiz_resources = []
            
            for rid, score in sorted_resources:
                # Lower threshold to include more resources
                if score < 0.01:  # Skip only very low scoring resources
                    continue
                    
                # Find the resource by ID
                resource = next((r for r in self.resources if r['id'] == rid), None)
                
                if not resource:
                    continue
                
                resource_with_score = {**resource, 'relevance_score': score}
                
                if resource['type'] == 'quiz':
                    quiz_resources.append(resource_with_score)
                elif resource['difficulty'] == 'beginner':
                    beginner_resources.append(resource_with_score)
                elif resource['difficulty'] == 'intermediate':
                    intermediate_resources.append(resource_with_score)
                else:
                    advanced_resources.append(resource_with_score)
            
            # FALLBACK: If no resources found by scoring, get by topic directly
            if not beginner_resources and not intermediate_resources:
                # Get all resources for this topic
                for resource in self.resources:
                    if resource.get('topic') == topic or topic in resource.get('tags', []):
                        resource_with_score = {**resource, 'relevance_score': 0.5}
                        if resource['type'] == 'quiz':
                            quiz_resources.append(resource_with_score)
                        elif resource.get('difficulty') == 'beginner':
                            beginner_resources.append(resource_with_score)
                        elif resource.get('difficulty') == 'intermediate':
                            intermediate_resources.append(resource_with_score)
                        else:
                            advanced_resources.append(resource_with_score)
            
            # Build the learning path
            # Step 1: Beginner article or video
            for r in beginner_resources:
                if r['type'] in ['article', 'video']:
                    learning_path.append({
                        'step': 1,
                        'title': 'Start Here',
                        'subtitle': 'Begin with the fundamentals',
                        'resource': r,
                        'reason': f"This {r['type']} provides a beginner-friendly introduction to help you build foundational understanding."
                    })
                    break
            
            # Step 2: Intermediate content
            for r in intermediate_resources:
                if r['type'] in ['article', 'video']:
                    learning_path.append({
                        'step': 2,
                        'title': 'Build Your Understanding',
                        'subtitle': 'Deepen your knowledge',
                        'resource': r,
                        'reason': f"This {r['type']} builds on the basics and introduces more advanced concepts."
                    })
                    break
            
            # If no intermediate, use an advanced resource
            if len(learning_path) < 2:
                for r in advanced_resources:
                    if r['type'] in ['article', 'video']:
                        learning_path.append({
                            'step': 2,
                            'title': 'Build Your Understanding',
                            'subtitle': 'Advance your knowledge',
                            'resource': r,
                            'reason': f"This {r['type']} provides deeper insights into the topic."
                        })
                        break
            
            # Step 3: Quiz - MUST match the topic!
            # Filter quizzes by topic first, then by relevance score
            topic_quizzes = [q for q in quiz_resources if q.get('topic') == topic]
            if not topic_quizzes:
                # Fallback: quizzes with topic in tags or subtopics
                topic_quizzes = [q for q in quiz_resources if topic in q.get('tags', []) or topic in q.get('subtopics', [])]
            if not topic_quizzes:
                # Last resort: any quiz with high relevance score
                topic_quizzes = [q for q in quiz_resources if q.get('relevance_score', 0) > 0.1]
            
            if topic_quizzes:
                # Sort by relevance score and take the best one
                topic_quizzes.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                learning_path.append({
                    'step': 3,
                    'title': 'Test Yourself',
                    'subtitle': 'Check your understanding',
                    'resource': topic_quizzes[0],
                    'reason': f"This quiz tests your understanding of {topic.replace('_', ' ').title()} and helps identify areas for review."
                })
            elif quiz_resources:
                # If no topic match, still add a quiz but note it's general
                learning_path.append({
                    'step': 3,
                    'title': 'Test Yourself',
                    'subtitle': 'Check your understanding',
                    'resource': quiz_resources[0],
                    'reason': "Take this quiz to validate your understanding and identify areas for review."
                })
            
            # ULTIMATE FALLBACK: If still no learning path, add ANY beginner resource
            if not learning_path and self.resources:
                for resource in self.resources:
                    if resource.get('difficulty') == 'beginner' and resource['type'] in ['article', 'video']:
                        learning_path.append({
                            'step': 1,
                            'title': 'Start Here',
                            'subtitle': 'A great place to begin your learning journey',
                            'resource': {**resource, 'relevance_score': 0.3},
                            'reason': f"This {resource['type']} is a helpful starting point for learning about technology topics."
                        })
                        break
            
            response['learning_path'] = learning_path
            
            # ===== GENERATE EXPLANATIONS =====
            # Generate topic explanation
            response['topic_explanation'] = generate_topic_explanation(
                topic,
                parsed_query['uganda_context']
            )
            
            # Generate Uganda relevance if context detected
            if parsed_query['uganda_context']['has_uganda_context'] or 'uganda' in query.lower():
                response['uganda_relevance'] = generate_uganda_relevance(
                    topic,
                    parsed_query['uganda_context']
                )
            else:
                response['uganda_relevance'] = None
            
            # ===== ADD ADDITIONAL RECOMMENDATIONS =====
            # Provide extra resources beyond the main path
            additional_recs = []
            seen_ids = {step['resource']['id'] for step in learning_path if 'resource' in step}
            
            for rid, score in sorted_resources[:10]:
                if rid not in seen_ids and score >= 0.01:
                    resource = next((r for r in self.resources if r['id'] == rid), None)
                    if resource:
                        additional_recs.append({
                            **resource,
                            'relevance_score': score
                        })
                        if len(additional_recs) >= 5:
                            break
            
            response['additional_recommendations'] = additional_recs
            
        except Exception as e:
            response['status'] = 'error'
            response['error'] = str(e)
        
        return response
    
    def add_feedback(self, resource_id: str, rating: int = None,
                     helpful: bool = None, comment: str = None,
                     query: str = None):
        """
        Add user feedback for a recommendation.
        
        This enables the LEARN pillar - feedback improves future recommendations.
        
        Args:
            resource_id: ID of the resource being rated
            rating: 1-5 star rating
            helpful: Was this helpful? (yes/no)
            comment: Optional text feedback
            query: Original query
        """
        self.feedback_manager.add_feedback(
            resource_id=resource_id,
            rating=rating,
            helpful=helpful,
            comment=comment,
            query=query
        )
    
    def get_system_stats(self) -> Dict:
        """
        Get system statistics for the dashboard.
        
        Returns:
            Dictionary with system statistics
        """
        kg_stats = self.knowledge_graph.get_statistics()
        
        return {
            'total_resources': len(self.resources),
            'total_topics': kg_stats['topics'],
            'total_applications': kg_stats['applications'],
            'graph_nodes': kg_stats['total_nodes'],
            'graph_edges': kg_stats['total_edges'],
            'feedback_count': self.feedback_manager.get_total_feedback_count(),
            'nlp_status': 'Loaded',
            'kg_status': 'Active',
            'recommender_status': 'Running',
            'feedback_status': 'Ready'
        }
    
    def get_resource_by_id(self, resource_id: str) -> Optional[Dict]:
        """
        Get a resource by its ID.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            Resource dictionary or None
        """
        for resource in self.resources:
            if resource['id'] == resource_id:
                return resource
        return None
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics."""
        return list(self.knowledge_graph.topics)
    
    def get_resources_by_topic(self, topic: str) -> List[Dict]:
        """Get all resources for a specific topic."""
        return self.knowledge_graph.get_resources_for_topic(topic)


# ==============================================================
# Evaluation Metrics
# ==============================================================

def evaluate_recommendations(recommender: CognitiveRecommender,
                            test_queries: List[Dict]) -> Dict:
    """
    Evaluate recommendation quality using test queries.
    
    Metrics:
    - Precision@K: How many top-K recommendations are relevant?
    - Recall: How many relevant items were recommended?
    - Topic Accuracy: Did we identify the correct topic?
    
    Args:
        recommender: The cognitive recommender instance
        test_queries: List of {query, expected_topic, relevant_resources}
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'precision_at_3': [],
        'precision_at_5': [],
        'topic_accuracy': [],
        'uganda_context_detection': []
    }
    
    for test in test_queries:
        query = test['query']
        expected_topic = test.get('expected_topic')
        expected_uganda = test.get('has_uganda_context', False)
        relevant_resources = set(test.get('relevant_resources', []))
        
        # Get recommendations
        result = recommender.process_query(query)
        
        # Check topic accuracy
        if expected_topic:
            detected_topic = result.get('parsed_query', {}).get('primary_topic')
            metrics['topic_accuracy'].append(1 if detected_topic == expected_topic else 0)
        
        # Check Uganda context detection
        detected_uganda = result.get('parsed_query', {}).get('uganda_context', {}).get('has_uganda_context', False)
        metrics['uganda_context_detection'].append(1 if detected_uganda == expected_uganda else 0)
        
        # Check precision if relevant resources provided
        if relevant_resources:
            learning_path = result.get('learning_path', [])
            recommended_ids = [step['resource']['id'] for step in learning_path if 'resource' in step]
            
            # Precision@3
            top_3 = recommended_ids[:3]
            relevant_in_top_3 = len(set(top_3) & relevant_resources)
            metrics['precision_at_3'].append(relevant_in_top_3 / min(3, len(relevant_resources)))
            
            # Precision@5
            additional = result.get('additional_recommendations', [])
            top_5_ids = recommended_ids + [r['id'] for r in additional][:5-len(recommended_ids)]
            relevant_in_top_5 = len(set(top_5_ids) & relevant_resources)
            metrics['precision_at_5'].append(relevant_in_top_5 / min(5, len(relevant_resources)))
    
    # Compute averages
    results = {}
    for metric, values in metrics.items():
        if values:
            results[metric] = sum(values) / len(values)
        else:
            results[metric] = 0.0
    
    results['total_queries'] = len(test_queries)
    
    return results


def compare_with_baseline(recommender: CognitiveRecommender,
                          test_queries: List[Dict]) -> Dict:
    """
    Compare cognitive recommender with a simple keyword baseline.
    
    Baseline: Simple keyword matching without NLP or knowledge graph.
    
    Args:
        recommender: The cognitive recommender
        test_queries: Test queries with expected results
        
    Returns:
        Comparison metrics
    """
    cognitive_results = []
    baseline_results = []
    
    for test in test_queries:
        query = test['query']
        relevant = set(test.get('relevant_resources', []))
        
        if not relevant:
            continue
        
        # Cognitive recommender
        result = recommender.process_query(query)
        cognitive_recs = [step['resource']['id'] for step in result.get('learning_path', []) if 'resource' in step]
        cognitive_precision = len(set(cognitive_recs[:3]) & relevant) / 3 if cognitive_recs else 0
        cognitive_results.append(cognitive_precision)
        
        # Baseline: Simple keyword matching
        query_words = set(query.lower().split())
        baseline_scores = {}
        
        for resource in recommender.resources:
            title_words = set(resource.get('title', '').lower().split())
            desc_words = set(resource.get('description', '').lower().split())
            all_words = title_words | desc_words
            
            overlap = len(query_words & all_words)
            baseline_scores[resource['id']] = overlap
        
        baseline_recs = sorted(baseline_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        baseline_ids = [r[0] for r in baseline_recs]
        baseline_precision = len(set(baseline_ids) & relevant) / 3 if baseline_ids else 0
        baseline_results.append(baseline_precision)
    
    return {
        'cognitive_avg_precision': sum(cognitive_results) / len(cognitive_results) if cognitive_results else 0,
        'baseline_avg_precision': sum(baseline_results) / len(baseline_results) if baseline_results else 0,
        'improvement': (sum(cognitive_results) - sum(baseline_results)) / len(baseline_results) if baseline_results else 0,
        'num_queries': len(cognitive_results)
    }


# ==============================================================
# Main execution for testing
# ==============================================================

if __name__ == "__main__":
    import os
    
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'educational_content.json')
    feedback_path = os.path.join(current_dir, '..', 'data', 'feedback.json')
    
    print("=" * 60)
    print("Brain Sparks - Cognitive Recommender Test")
    print("=" * 60)
    
    # Initialize system
    recommender = CognitiveRecommender(data_path, feedback_path)
    
    # Test query
    test_query = "Explain the basics of quantum computing and show me how it could be relevant for solving problems in Uganda."
    
    print(f"\n Processing query:")
    print(f"   '{test_query}'")
    print("-" * 60)
    
    # Get recommendations
    result = recommender.process_query(test_query)
    
    # Display results
    print(f"\n Topic Identified: {result['parsed_query']['primary_topic']}")
    print(f"   Confidence: {result['parsed_query']['topic_confidence']:.2f}")
    print(f"\n Uganda Context: {result['parsed_query']['uganda_context']['has_uganda_context']}")
    
    print(f"\n Learning Path:")
    for step in result.get('learning_path', []):
        resource = step.get('resource', {})
        print(f"\n   Step {step['step']}: {step['title']}")
        print(f"    {resource.get('title', 'Unknown')}")
        print(f"   Type: {resource.get('type', 'N/A')} | Difficulty: {resource.get('difficulty', 'N/A')}")
        print(f"    {step.get('reason', '')[:100]}")
    
    # Show system stats
    stats = recommender.get_system_stats()
    print(f"\n System Statistics:")
    print(f"   Resources: {stats['total_resources']}")
    print(f"   Topics: {stats['total_topics']}")
    print(f"   Feedback entries: {stats['feedback_count']}")
    
    print("\n Cognitive recommender test complete!")


