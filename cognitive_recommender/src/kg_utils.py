"""
============================================================
Brain Sparks - Knowledge Graph Utilities Module
============================================================
This module implements the REASON pillar of our cognitive system.

The REASON pillar is responsible for:
1. Building and maintaining a knowledge graph of educational content
2. Representing relationships between topics, resources, and applications
3. Enabling graph-based reasoning and querying
4. Supporting the recommendation engine with structured knowledge

What is a Knowledge Graph?
- A graph is a data structure with NODES (things) and EDGES (relationships)
- Nodes represent: Topics, Resources (articles/videos/quizzes), Applications
- Edges represent: "is_about", "requires", "leads_to", "applies_to"
- This structure allows us to "reason" about content relationships

Author: Rugogamu Noela
Institution: Uganda Christian University (UCU)
============================================================
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx

# ==============================================================
# Knowledge Graph Class
# ==============================================================

class EducationalKnowledgeGraph:
    """
    A knowledge graph for educational content and their relationships.
    
    The graph contains three types of nodes:
    1. TOPIC nodes: Educational topics (e.g., "quantum_computing", "machine_learning")
    2. RESOURCE nodes: Educational resources (articles, videos, quizzes)
    3. APPLICATION nodes: Real-world applications (e.g., "agriculture", "healthcare")
    
    Edges represent relationships:
    - "is_about": Resource -> Topic (a resource covers a topic)
    - "requires": Resource -> Resource (prerequisite relationship)
    - "leads_to": Topic -> Topic (learning progression)
    - "applies_to": Topic -> Application (topic has application in domain)
    - "relevant_for": Resource -> Context (resource is relevant for Uganda)
    """
    
    def __init__(self):
        """
        Initialize an empty knowledge graph.
        
        We use NetworkX, a powerful Python library for graph analysis.
        NetworkX provides:
        - Efficient graph data structures
        - Many graph algorithms (shortest path, centrality, etc.)
        - Easy visualization capabilities
        """
        # Create a directed graph (edges have direction, e.g., A -> B)
        self.graph = nx.DiGraph()
        
        # Store node metadata separately for quick access
        self.node_metadata = {}
        
        # Keep track of node types for filtering
        self.topics = set()
        self.resources = set()
        self.applications = set()
        
        # Index for fast lookups
        self.topic_resources = defaultdict(list)  # topic -> [resource_ids]
        self.resource_topics = defaultdict(list)  # resource_id -> [topics]
        
    def add_topic(self, topic_id: str, name: str, description: str = "", 
                  parent_topic: str = None):
        """
        Add a topic node to the knowledge graph.
        
        Topics are the main educational categories (e.g., "quantum_computing").
        
        Args:
            topic_id: Unique identifier for the topic
            name: Human-readable name
            description: Brief description of the topic
            parent_topic: Parent topic for hierarchical organization
        """
        # Add node to graph with type attribute
        self.graph.add_node(
            topic_id,
            node_type='topic',
            name=name,
            description=description
        )
        
        # Track this as a topic
        self.topics.add(topic_id)
        
        # Store metadata
        self.node_metadata[topic_id] = {
            'type': 'topic',
            'name': name,
            'description': description
        }
        
        # If there's a parent topic, create a "part_of" edge
        if parent_topic and parent_topic in self.topics:
            self.graph.add_edge(topic_id, parent_topic, relationship='part_of')
    
    def add_resource(self, resource_id: str, title: str, description: str,
                     resource_type: str, difficulty: str, topic: str,
                     metadata: Dict = None):
        """
        Add a resource node to the knowledge graph.
        
        Resources are educational materials (articles, videos, quizzes).
        
        Args:
            resource_id: Unique identifier (e.g., "QC001")
            title: Resource title
            description: Resource description
            resource_type: "article", "video", or "quiz"
            difficulty: "beginner", "intermediate", or "advanced"
            topic: Main topic this resource covers
            metadata: Additional metadata (url, duration, etc.)
        """
        # Add node to graph
        self.graph.add_node(
            resource_id,
            node_type='resource',
            title=title,
            description=description,
            resource_type=resource_type,
            difficulty=difficulty,
            topic=topic
        )
        
        # Track this as a resource
        self.resources.add(resource_id)
        
        # Store full metadata
        self.node_metadata[resource_id] = {
            'type': 'resource',
            'title': title,
            'description': description,
            'resource_type': resource_type,
            'difficulty': difficulty,
            'topic': topic,
            **(metadata or {})
        }
        
        # Create "is_about" edge from resource to topic
        if topic in self.topics:
            self.graph.add_edge(resource_id, topic, relationship='is_about')
            
        # Update indexes
        self.topic_resources[topic].append(resource_id)
        self.resource_topics[resource_id].append(topic)
    
    def add_application(self, app_id: str, name: str, description: str,
                        related_topics: List[str] = None):
        """
        Add an application domain node to the knowledge graph.
        
        Applications represent real-world domains where topics can be applied
        (e.g., "agriculture", "healthcare", "finance").
        
        Args:
            app_id: Unique identifier
            name: Application domain name
            description: Description of the application domain
            related_topics: Topics that apply to this domain
        """
        # Add node
        self.graph.add_node(
            app_id,
            node_type='application',
            name=name,
            description=description
        )
        
        # Track as application
        self.applications.add(app_id)
        
        # Store metadata
        self.node_metadata[app_id] = {
            'type': 'application',
            'name': name,
            'description': description
        }
        
        # Create "applies_to" edges from topics to this application
        if related_topics:
            for topic in related_topics:
                if topic in self.topics:
                    self.graph.add_edge(topic, app_id, relationship='applies_to')
    
    def add_prerequisite(self, resource_id: str, prerequisite_id: str):
        """
        Add a prerequisite relationship between resources.
        
        This represents that one resource should be completed before another.
        
        Args:
            resource_id: The resource that has a prerequisite
            prerequisite_id: The prerequisite resource
        """
        if resource_id in self.resources and prerequisite_id in self.resources:
            self.graph.add_edge(
                prerequisite_id, 
                resource_id, 
                relationship='leads_to'
            )
    
    def add_topic_relationship(self, from_topic: str, to_topic: str, 
                               relationship: str = 'related_to'):
        """
        Add a relationship between topics.
        
        Args:
            from_topic: Source topic
            to_topic: Target topic
            relationship: Type of relationship (e.g., "leads_to", "related_to")
        """
        if from_topic in self.topics and to_topic in self.topics:
            self.graph.add_edge(from_topic, to_topic, relationship=relationship)
    
    def get_resources_for_topic(self, topic: str, 
                                 difficulty: str = None,
                                 resource_type: str = None) -> List[Dict]:
        """
        Get all resources related to a topic.
        
        This is a key method for the recommendation system.
        
        Args:
            topic: Topic ID to find resources for
            difficulty: Optional filter by difficulty level
            resource_type: Optional filter by resource type
            
        Returns:
            List of resource dictionaries
        """
        resources = []
        
        # Get all resource IDs for this topic
        resource_ids = self.topic_resources.get(topic, [])
        
        for rid in resource_ids:
            if rid in self.node_metadata:
                resource = self.node_metadata[rid].copy()
                resource['id'] = rid
                
                # Apply filters
                if difficulty and resource.get('difficulty') != difficulty:
                    continue
                if resource_type and resource.get('resource_type') != resource_type:
                    continue
                
                resources.append(resource)
        
        return resources
    
    def get_related_topics(self, topic: str, max_depth: int = 2) -> List[str]:
        """
        Get topics related to a given topic using graph traversal.
        
        Uses breadth-first search to find connected topics.
        
        Args:
            topic: Starting topic
            max_depth: Maximum number of edges to traverse
            
        Returns:
            List of related topic IDs
        """
        if topic not in self.topics:
            return []
        
        related = set()
        
        # BFS traversal
        visited = {topic}
        queue = [(topic, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors (both directions for undirected relationships)
            neighbors = set(self.graph.successors(current)) | set(self.graph.predecessors(current))
            
            for neighbor in neighbors:
                if neighbor in self.topics and neighbor not in visited:
                    related.add(neighbor)
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return list(related)
    
    def get_learning_path(self, resource_id: str) -> List[str]:
        """
        Get the learning path (prerequisites) for a resource.
        
        Traverses "leads_to" edges backwards to find what should be
        learned before this resource.
        
        Args:
            resource_id: Target resource ID
            
        Returns:
            Ordered list of resource IDs (prerequisites first)
        """
        if resource_id not in self.resources:
            return [resource_id] if resource_id else []
        
        path = []
        visited = set()
        
        def find_prerequisites(rid):
            """Recursively find all prerequisites."""
            if rid in visited:
                return
            visited.add(rid)
            
            # Find predecessors with "leads_to" relationship
            for pred in self.graph.predecessors(rid):
                edge_data = self.graph.get_edge_data(pred, rid)
                if edge_data and edge_data.get('relationship') == 'leads_to':
                    find_prerequisites(pred)
            
            path.append(rid)
        
        find_prerequisites(resource_id)
        return path
    
    def get_applications_for_topic(self, topic: str) -> List[Dict]:
        """
        Get real-world applications for a topic.
        
        Args:
            topic: Topic ID
            
        Returns:
            List of application dictionaries
        """
        applications = []
        
        if topic not in self.topics:
            return applications
        
        # Find all application nodes connected to this topic
        for neighbor in self.graph.successors(topic):
            edge_data = self.graph.get_edge_data(topic, neighbor)
            if edge_data and edge_data.get('relationship') == 'applies_to':
                if neighbor in self.node_metadata:
                    app_info = self.node_metadata[neighbor].copy()
                    app_info['id'] = neighbor
                    applications.append(app_info)
        
        return applications
    
    def compute_topic_importance(self) -> Dict[str, float]:
        """
        Compute importance scores for topics using PageRank.
        
        PageRank is an algorithm originally used by Google to rank web pages.
        Topics that are connected to many other important topics get higher scores.
        
        Returns:
            Dictionary mapping topic IDs to importance scores
        """
        # Get subgraph of only topic nodes
        topic_subgraph = self.graph.subgraph(self.topics)
        
        if len(topic_subgraph) == 0:
            return {}
        
        try:
            # Compute PageRank
            pagerank = nx.pagerank(topic_subgraph, alpha=0.85)
            return pagerank
        except:
            # If PageRank fails, return equal scores
            return {t: 1.0 / len(self.topics) for t in self.topics}
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'topics': len(self.topics),
            'resources': len(self.resources),
            'applications': len(self.applications),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
    
    def to_dict(self) -> Dict:
        """
        Export graph to dictionary format.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': [(u, v, d) for u, v, d in self.graph.edges(data=True)],
            'metadata': self.node_metadata,
            'topics': list(self.topics),
            'resources': list(self.resources),
            'applications': list(self.applications)
        }


# ==============================================================
# Entity Extraction and Relationship Identification
# ==============================================================

def extract_entity_attributes(topic: str, resources: List[Dict], 
                              topic_hierarchy: Dict = None) -> Dict:
    """
    Extract meaningful attributes for a topic entity from resources.
    
    Attributes include:
    - Definition: What the topic is
    - Category/Type: Classification of the topic
    - Examples: Concrete examples from resources
    - Inputs/Outputs: What the topic takes as input and produces
    - Difficulty level: Range of difficulty levels available
    - Domain relevance: Uganda-specific relevance
    
    Args:
        topic: Topic identifier
        resources: List of all resources
        topic_hierarchy: Optional topic hierarchy data
        
    Returns:
        Dictionary with entity attributes
    """
    # Filter resources for this topic
    topic_resources = [r for r in resources if r.get('topic') == topic]
    
    # Extract definition from descriptions
    definitions = []
    for r in topic_resources:
        desc = r.get('description', '')
        if desc:
            # Take first sentence as definition
            first_sent = desc.split('.')[0] if '.' in desc else desc[:100]
            definitions.append(first_sent)
    
    definition = definitions[0] if definitions else f"Educational topic: {topic.replace('_', ' ').title()}"
    
    # Extract examples from subtopics and tags
    examples = set()
    for r in topic_resources:
        examples.update(r.get('subtopics', []))
        examples.update([tag for tag in r.get('tags', []) if tag != topic])
    
    # Determine category/type
    category_keywords = {
        'programming': ['programming', 'coding', 'software', 'development'],
        'data': ['data', 'analysis', 'analytics', 'science'],
        'ai_ml': ['machine learning', 'artificial intelligence', 'neural', 'deep learning'],
        'web': ['web', 'frontend', 'backend', 'fullstack'],
        'security': ['security', 'cybersecurity', 'encryption', 'protection'],
        'cloud': ['cloud', 'aws', 'azure', 'gcp', 'infrastructure'],
        'mobile': ['mobile', 'android', 'ios', 'app'],
        'blockchain': ['blockchain', 'crypto', 'distributed', 'ledger'],
        'iot': ['iot', 'internet of things', 'sensors', 'embedded'],
        'health': ['health', 'medical', 'healthcare', 'diagnosis'],
        'finance': ['finance', 'fintech', 'banking', 'payment'],
        'education': ['education', 'learning', 'teaching', 'pedagogy'],
        'ethics': ['ethics', 'governance', 'responsible', 'fairness']
    }
    
    topic_lower = topic.lower()
    category = 'general'
    for cat, keywords in category_keywords.items():
        if any(kw in topic_lower for kw in keywords):
            category = cat
            break
    
    # Extract difficulty levels
    difficulties = set(r.get('difficulty', 'beginner') for r in topic_resources)
    difficulty_range = sorted(list(difficulties), key=lambda x: ['beginner', 'intermediate', 'advanced'].index(x) if x in ['beginner', 'intermediate', 'advanced'] else 0)
    
    # Extract Uganda relevance
    uganda_relevance = set()
    for r in topic_resources:
        uganda_relevance.update(r.get('uganda_relevance', []))
        if r.get('uganda_applications'):
            uganda_relevance.add('applicable')
    
    # Extract inputs/outputs from learning outcomes and descriptions
    inputs = set()
    outputs = set()
    for r in topic_resources:
        outcomes = r.get('learning_outcomes', [])
        for outcome in outcomes:
            # Extract what you need (inputs)
            if 'understand' in outcome.lower() or 'learn' in outcome.lower():
                inputs.add('knowledge')
            if 'python' in outcome.lower() or 'code' in outcome.lower():
                inputs.add('programming_skills')
            if 'data' in outcome.lower():
                inputs.add('data')
            
            # Extract what you produce (outputs)
            if 'build' in outcome.lower() or 'create' in outcome.lower():
                outputs.add('applications')
            if 'analyze' in outcome.lower() or 'predict' in outcome.lower():
                outputs.add('insights')
            if 'implement' in outcome.lower():
                outputs.add('solutions')
    
    return {
        'definition': definition,
        'category': category,
        'examples': list(examples)[:10],  # Limit to 10 examples
        'inputs': list(inputs) if inputs else ['general_knowledge'],
        'outputs': list(outputs) if outputs else ['knowledge'],
        'difficulty_levels': difficulty_range,
        'domain_relevance': list(uganda_relevance),
        'resource_count': len(topic_resources),
        'subtopics': list(set().union(*[r.get('subtopics', []) for r in topic_resources]))[:15]
    }


def extract_relationships(topic: str, all_topics: Set[str], resources: List[Dict],
                         topic_hierarchy: Dict = None) -> List[Tuple[str, str, str]]:
    """
    Extract meaningful relationships between topics.
    
    Relationship types:
    - has_subtopic: Topic contains subtopic
    - part_of: Topic is part of a larger topic
    - type_of: Topic is a type/category of another
    - used_for: Topic is used for another topic/application
    - improves: Topic improves/enhances another
    - requires: Topic requires another as prerequisite
    - produces: Topic produces/creates another
    - relates_to: General relationship
    - depends_on: Topic depends on another
    - enables: Topic enables another
    
    Args:
        topic: Source topic
        all_topics: Set of all available topics
        resources: List of all resources
        topic_hierarchy: Optional topic hierarchy
        
    Returns:
        List of (from_topic, to_topic, relationship_type) tuples
    """
    relationships = []
    topic_resources = [r for r in resources if r.get('topic') == topic]
    
    # 1. HAS_SUBTOPIC relationships
    for r in topic_resources:
        for subtopic in r.get('subtopics', []):
            if subtopic in all_topics and subtopic != topic:
                relationships.append((topic, subtopic, 'has_subtopic'))
    
    # 2. PART_OF relationships (from topic hierarchy and logical groupings)
    part_of_mappings = {
        'machine_learning': 'artificial_intelligence',
        'deep_learning': 'machine_learning',
        'neural_networks': 'machine_learning',
        'supervised_learning': 'machine_learning',
        'unsupervised_learning': 'machine_learning',
        'natural_language_processing': 'artificial_intelligence',
        'computer_vision': 'artificial_intelligence',
        'data_science': 'machine_learning',
        'classification': 'machine_learning',
        'regression': 'machine_learning',
        'frontend': 'web_development',
        'backend': 'web_development',
        'fullstack': 'web_development',
        'react': 'web_development',
        'nodejs': 'web_development',
        'python': 'programming',
        'javascript': 'programming',
        'java': 'programming',
        'sql': 'databases',
        'nosql': 'databases',
        'mongodb': 'databases',
        'postgresql': 'databases'
    }
    
    if topic in part_of_mappings:
        parent = part_of_mappings[topic]
        if parent in all_topics:
            relationships.append((topic, parent, 'part_of'))
    
    # 3. REQUIRES/DEPENDS_ON relationships (from prerequisites)
    for r in topic_resources:
        prereqs = r.get('prerequisites', [])
        for prereq_id in prereqs:
            # Find the topic of the prerequisite resource
            for other_r in resources:
                if other_r.get('id') == prereq_id:
                    prereq_topic = other_r.get('topic')
                    if prereq_topic in all_topics and prereq_topic != topic:
                        relationships.append((topic, prereq_topic, 'requires'))
                        relationships.append((topic, prereq_topic, 'depends_on'))
                    break
    
    # 4. USED_FOR relationships (from Uganda applications and tags)
    for r in topic_resources:
        uganda_apps = r.get('uganda_applications', '')
        if 'agriculture' in uganda_apps.lower():
            if 'agriculture' in all_topics:
                relationships.append((topic, 'agriculture', 'used_for'))
        if 'healthcare' in uganda_apps.lower() or 'health' in uganda_apps.lower():
            if 'digital_health' in all_topics:
                relationships.append((topic, 'digital_health', 'used_for'))
        if 'finance' in uganda_apps.lower() or 'financial' in uganda_apps.lower():
            if 'fintech' in all_topics:
                relationships.append((topic, 'fintech', 'used_for'))
    
    # 5. IMPROVES relationships (from descriptions and learning outcomes)
    improvement_keywords = {
        'optimize': 'optimization',
        'enhance': 'performance',
        'improve': 'efficiency',
        'accelerate': 'speed',
        'secure': 'security'
    }
    
    for r in topic_resources:
        desc = r.get('description', '').lower()
        for keyword, related_concept in improvement_keywords.items():
            if keyword in desc:
                # Try to find related topics
                for other_topic in all_topics:
                    if related_concept in other_topic.lower():
                        relationships.append((topic, other_topic, 'improves'))
                        break
    
    # 6. ENABLES relationships (logical connections)
    enables_mappings = {
        'programming': ['web_development', 'mobile_development', 'data_science', 'machine_learning'],
        'databases': ['web_development', 'data_science'],
        'cloud_computing': ['web_development', 'machine_learning'],
        'machine_learning': ['digital_health', 'fintech', 'agriculture'],
        'artificial_intelligence': ['digital_health', 'education'],
        'blockchain': ['fintech'],
        'cybersecurity': ['fintech', 'web_development'],
        'data_science': ['machine_learning', 'digital_health']
    }
    
    if topic in enables_mappings:
        for enabled_topic in enables_mappings[topic]:
            if enabled_topic in all_topics:
                relationships.append((topic, enabled_topic, 'enables'))
    
    # 7. RELATES_TO relationships (from tags and secondary topics)
    for r in topic_resources:
        tags = r.get('tags', [])
        for tag in tags:
            # Check if tag matches another topic
            for other_topic in all_topics:
                if tag.lower() in other_topic.lower() or other_topic.lower() in tag.lower():
                    if other_topic != topic and (topic, other_topic, 'relates_to') not in relationships:
                        relationships.append((topic, other_topic, 'relates_to'))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_relationships = []
    for rel in relationships:
        if rel not in seen:
            seen.add(rel)
            unique_relationships.append(rel)
    
    return unique_relationships


def build_knowledge_graph_from_data(data_path: str) -> EducationalKnowledgeGraph:
    """
    Build a knowledge graph from the educational content JSON file.
    
    This function:
    1. Loads the educational content data
    2. Creates topic nodes for all unique topics
    3. Creates resource nodes for all educational materials
    4. Creates application nodes for Uganda-relevant domains
    5. Establishes relationships between nodes
    
    Args:
        data_path: Path to educational_content.json
        
    Returns:
        Populated EducationalKnowledgeGraph
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    resources = data.get('resources', [])
    topic_hierarchy = data.get('topic_hierarchy', {})
    uganda_tags = data.get('uganda_tags', {})
    
    # Create knowledge graph
    kg = EducationalKnowledgeGraph()
    
    # ===== Step 1: Extract all topics =====
    # Collect all unique topics from resources
    all_topics = set()
    for resource in resources:
        all_topics.add(resource.get('topic', 'general'))
        all_topics.update(resource.get('subtopics', []))
    
    # ===== Step 2: Extract entities with attributes =====
    # Extract meaningful entities with rich attributes
    topic_names = {
        'quantum_computing': 'Quantum Computing',
        'machine_learning': 'Machine Learning',
        'data_science': 'Data Science',
        'artificial_intelligence': 'Artificial Intelligence',
        'web_development': 'Web Development',
        'cybersecurity': 'Cybersecurity',
        'cloud_computing': 'Cloud Computing',
        'databases': 'Databases',
        'blockchain': 'Blockchain',
        'internet_of_things': 'Internet of Things',
        'digital_health': 'Digital Health',
        'fintech': 'Financial Technology',
        'programming': 'Programming',
        'educational_technology': 'Educational Technology',
        'environmental_tech': 'Environmental Technology',
        'entrepreneurship': 'Entrepreneurship',
        'project_management': 'Project Management',
        'design_thinking': 'Design Thinking',
        'technology_ethics': 'Technology Ethics',
        'mobile_development': 'Mobile Development',
        'devops': 'DevOps',
        'software_quality': 'Software Quality'
    }
    
    # Extract entity attributes for each topic
    entity_attributes = {}
    for topic in all_topics:
        name = topic_names.get(topic, topic.replace('_', ' ').title())
        attributes = extract_entity_attributes(topic, resources, topic_hierarchy)
        entity_attributes[topic] = attributes
        
        # Add topic node with rich attributes
        kg.add_topic(topic, name, attributes.get('definition', f"Educational resources about {name}"))
        
        # Store additional attributes in node metadata
        kg.node_metadata[topic].update({
            'category': attributes.get('category', 'general'),
            'examples': attributes.get('examples', []),
            'inputs': attributes.get('inputs', []),
            'outputs': attributes.get('outputs', []),
            'difficulty_levels': attributes.get('difficulty_levels', []),
            'domain_relevance': attributes.get('domain_relevance', []),
            'resource_count': attributes.get('resource_count', 0),
            'subtopics': attributes.get('subtopics', [])
        })
        
        # Update graph node with attributes
        kg.graph.nodes[topic].update({
            'category': attributes.get('category', 'general'),
            'examples': attributes.get('examples', []),
            'difficulty_levels': attributes.get('difficulty_levels', []),
            'domain_relevance': attributes.get('domain_relevance', [])
        })
    
    # ===== Step 3: Extract and add relationships =====
    # Extract meaningful relationships between entities
    all_relationships = []
    for topic in all_topics:
        relationships = extract_relationships(topic, all_topics, resources, topic_hierarchy)
        all_relationships.extend(relationships)
    
    # Add all extracted relationships
    for from_topic, to_topic, rel_type in all_relationships:
        if from_topic in kg.topics and to_topic in kg.topics:
            kg.add_topic_relationship(from_topic, to_topic, rel_type)
    
    # ===== Step 4: Add application domains =====
    for app_id, description in uganda_tags.items():
        kg.add_application(
            f"app_{app_id}",
            app_id.replace('_', ' ').title(),
            description
        )
    
    # Connect applications to relevant topics
    topic_applications = {
        'machine_learning': ['app_agriculture', 'app_healthcare', 'app_finance', 'app_education'],
        'artificial_intelligence': ['app_healthcare', 'app_government', 'app_education'],
        'data_science': ['app_government', 'app_healthcare', 'app_finance'],
        'internet_of_things': ['app_agriculture', 'app_energy', 'app_healthcare'],
        'fintech': ['app_finance'],
        'digital_health': ['app_healthcare'],
        'cybersecurity': ['app_government', 'app_finance'],
        'web_development': ['app_government', 'app_education'],
        'blockchain': ['app_finance', 'app_government'],
        'environmental_tech': ['app_environment', 'app_agriculture']
    }
    
    for topic, apps in topic_applications.items():
        if topic in kg.topics:
            for app in apps:
                if app in kg.applications:
                    kg.graph.add_edge(topic, app, relationship='applies_to')
    
    # ===== Step 5: Add all resources =====
    for resource in resources:
        resource_id = resource.get('id')
        
        kg.add_resource(
            resource_id=resource_id,
            title=resource.get('title', ''),
            description=resource.get('description', ''),
            resource_type=resource.get('type', 'article'),
            difficulty=resource.get('difficulty', 'beginner'),
            topic=resource.get('topic', 'general'),
            metadata={
                'duration_minutes': resource.get('duration_minutes', 0),
                'source': resource.get('source', ''),
                'url': resource.get('url', ''),
                'uganda_relevance': resource.get('uganda_relevance', []),
                'uganda_applications': resource.get('uganda_applications', ''),
                'tags': resource.get('tags', []),
                'learning_outcomes': resource.get('learning_outcomes', []),
                'subtopics': resource.get('subtopics', []),
                'prerequisites': resource.get('prerequisites', []),
                'questions': resource.get('questions', [])
            }
        )
        
        # Add edges to subtopics
        for subtopic in resource.get('subtopics', []):
            if subtopic in kg.topics:
                kg.graph.add_edge(resource_id, subtopic, relationship='covers')
        
        # Add Uganda relevance edges
        for relevance in resource.get('uganda_relevance', []):
            app_id = f"app_{relevance}"
            if app_id in kg.applications:
                kg.graph.add_edge(resource_id, app_id, relationship='relevant_for')
    
    # ===== Step 6: Add prerequisite relationships =====
    for resource in resources:
        resource_id = resource.get('id')
        for prereq in resource.get('prerequisites', []):
            kg.add_prerequisite(resource_id, prereq)
    
    return kg


def get_topic_subgraph(kg: EducationalKnowledgeGraph, topic: str, 
                       depth: int = 2) -> nx.DiGraph:
    """
    Extract a subgraph centered on a specific topic.
    
    Useful for visualization and focused analysis.
    
    Args:
        kg: The knowledge graph
        topic: Central topic
        depth: How many edges to include from center
        
    Returns:
        NetworkX subgraph
    """
    if topic not in kg.graph:
        return nx.DiGraph()
    
    # Get nodes within depth using BFS
    nodes = set([topic])
    frontier = set([topic])
    
    for _ in range(depth):
        new_frontier = set()
        for node in frontier:
            neighbors = set(kg.graph.successors(node)) | set(kg.graph.predecessors(node))
            new_frontier.update(neighbors - nodes)
        nodes.update(new_frontier)
        frontier = new_frontier
    
    return kg.graph.subgraph(nodes).copy()


def create_pyvis_graph(kg: EducationalKnowledgeGraph, topic: str = None, 
                       output_path: str = "knowledge_graph.html") -> str:
    """
    Create an interactive knowledge graph visualization using PyVis.
    
    Args:
        kg: The knowledge graph
        topic: Optional topic to filter the graph
        output_path: Path to save the HTML file
        
    Returns:
        Path to the generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("PyVis is required. Install with: pip install pyvis")
    
    # Get subgraph if topic is specified
    if topic and topic in kg.graph:
        subgraph = get_topic_subgraph(kg, topic, depth=2)
    else:
        subgraph = kg.graph
    
    # Create PyVis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1a1a1a",
        directed=True,
        notebook=False
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 200,
          "springConstant": 0.01,
          "nodeRepulsion": 100,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion"
      }
    }
    """)
    
    # Add nodes with attributes
    for node_id in subgraph.nodes():
        node_data = subgraph.nodes[node_id]
        node_type = node_data.get('node_type', 'unknown')
        
        # Determine node properties based on type
        if node_type == 'topic':
            color = '#C41E3A'  # Red
            shape = 'dot'
            size = 25
            title = f"""
            <b>{node_data.get('name', node_id)}</b><br>
            <b>Type:</b> Topic<br>
            <b>Category:</b> {node_data.get('category', 'N/A')}<br>
            <b>Definition:</b> {node_data.get('description', 'N/A')[:100]}...<br>
            <b>Examples:</b> {', '.join(node_data.get('examples', [])[:3])}<br>
            <b>Difficulty Levels:</b> {', '.join(node_data.get('difficulty_levels', []))}<br>
            <b>Domain Relevance:</b> {', '.join(node_data.get('domain_relevance', [])[:3])}<br>
            <b>Resources:</b> {node_data.get('resource_count', 0)}
            """
        elif node_type == 'resource':
            color = '#2563EB'  # Blue
            shape = 'square'
            size = 15
            title = f"""
            <b>{node_data.get('title', node_id)}</b><br>
            <b>Type:</b> {node_data.get('resource_type', 'resource')}<br>
            <b>Difficulty:</b> {node_data.get('difficulty', 'N/A')}<br>
            <b>Description:</b> {node_data.get('description', 'N/A')[:100]}...
            """
        elif node_type == 'application':
            color = '#16A34A'  # Green
            shape = 'diamond'
            size = 20
            title = f"""
            <b>{node_data.get('name', node_id)}</b><br>
            <b>Type:</b> Application Domain<br>
            <b>Description:</b> {node_data.get('description', 'N/A')[:100]}...
            """
        else:
            color = '#6B7280'  # Gray
            shape = 'dot'
            size = 15
            title = node_id
        
        label = node_data.get('name', node_data.get('title', node_id))
        # Truncate long labels
        if len(label) > 20:
            label = label[:17] + "..."
        
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            shape=shape,
            size=size
        )
    
    # Add edges with relationship labels
    for u, v, data in subgraph.edges(data=True):
        relationship = data.get('relationship', 'related')
        
        # Color edges based on relationship type
        edge_colors = {
            'has_subtopic': '#C41E3A',
            'part_of': '#DC2626',
            'type_of': '#991B1B',
            'used_for': '#16A34A',
            'improves': '#059669',
            'requires': '#2563EB',
            'depends_on': '#1D4ED8',
            'produces': '#7C3AED',
            'relates_to': '#6B7280',
            'enables': '#10B981',
            'is_about': '#3B82F6',
            'covers': '#60A5FA',
            'applies_to': '#16A34A',
            'relevant_for': '#10B981'
        }
        
        edge_color = edge_colors.get(relationship, '#9CA3AF')
        
        net.add_edge(
            u, v,
            label=relationship.replace('_', ' ').title(),
            color=edge_color,
            width=2,
            arrows='to'
        )
    
    # Save the graph
    net.save_graph(output_path)
    return output_path


def visualize_graph_as_dict(kg: EducationalKnowledgeGraph, topic: str = None) -> Dict:
    """
    Convert knowledge graph to a format suitable for visualization.
    
    Returns a dictionary with nodes and edges that can be used
    with JavaScript visualization libraries or matplotlib.
    
    Args:
        kg: The knowledge graph
        topic: Optional topic to center visualization on
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
    """
    if topic:
        subgraph = get_topic_subgraph(kg, topic)
    else:
        subgraph = kg.graph
    
    # Prepare nodes
    nodes = []
    for node_id in subgraph.nodes():
        node_data = dict(subgraph.nodes[node_id])
        node_type = node_data.get('node_type', 'unknown')
        
        nodes.append({
            'id': node_id,
            'label': node_data.get('name', node_data.get('title', node_id)),
            'type': node_type,
            'color': {
                'topic': '#C41E3A',      # Red for topics
                'resource': '#2563EB',   # Blue for resources
                'application': '#16A34A' # Green for applications
            }.get(node_type, '#6B7280')
        })
    
    # Prepare edges
    edges = []
    for u, v, data in subgraph.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'relationship': data.get('relationship', 'related'),
            'label': data.get('relationship', '')
        })
    
    return {'nodes': nodes, 'edges': edges}


# ==============================================================
# Reasoning Functions
# These functions implement reasoning over the knowledge graph
# ==============================================================

def reason_about_query(kg: EducationalKnowledgeGraph, 
                       parsed_query: Dict) -> Dict:
    """
    Use the knowledge graph to reason about a user's query.
    
    This is the core reasoning function that:
    1. Maps the identified topic to resources
    2. Considers difficulty preferences
    3. Incorporates Uganda context
    4. Determines a logical learning order
    
    Args:
        kg: The knowledge graph
        parsed_query: Parsed query from QueryParser
        
    Returns:
        Dictionary with reasoning results
    """
    results = {
        'main_topic': parsed_query['primary_topic'],
        'topic_in_graph': parsed_query['primary_topic'] in kg.topics,
        'available_resources': [],
        'recommended_sequence': [],
        'related_topics': [],
        'uganda_applications': [],
        'reasoning_path': []
    }
    
    topic = parsed_query['primary_topic']
    
    # Step 1: Check if topic exists in our knowledge graph
    if topic in kg.topics:
        results['reasoning_path'].append(f"✓ Found topic '{topic}' in knowledge graph")
        
        # Step 2: Get all resources for this topic
        resources = kg.get_resources_for_topic(topic)
        results['available_resources'] = resources
        results['reasoning_path'].append(f"✓ Found {len(resources)} resources for this topic")
        
        # Step 3: Get related topics for broader learning
        related = kg.get_related_topics(topic)
        results['related_topics'] = related
        if related:
            results['reasoning_path'].append(f"✓ Found {len(related)} related topics: {related[:3]}")
        
        # Step 4: Get Uganda applications
        applications = kg.get_applications_for_topic(topic)
        results['uganda_applications'] = applications
        if applications:
            results['reasoning_path'].append(f"✓ Found {len(applications)} Uganda application domains")
    else:
        results['reasoning_path'].append(f"WARNING: Topic '{topic}' not directly in graph, searching related content")
    
    # Step 5: Determine recommended sequence based on difficulty
    if results['available_resources']:
        # Sort by difficulty
        difficulty_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        sorted_resources = sorted(
            results['available_resources'],
            key=lambda x: difficulty_order.get(x.get('difficulty', 'beginner'), 0)
        )
        results['recommended_sequence'] = [r.get('id') for r in sorted_resources]
        results['reasoning_path'].append("✓ Created learning sequence based on difficulty progression")
    
    return results


# ==============================================================
# Main execution for testing
# ==============================================================

if __name__ == "__main__":
    import os
    
    # Get the path to data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'educational_content.json')
    
    print("=" * 60)
    print("Brain Sparks - Knowledge Graph Builder Test")
    print("=" * 60)
    
    # Build knowledge graph
    print("\n Building knowledge graph from data...")
    kg = build_knowledge_graph_from_data(data_path)
    
    # Show statistics
    stats = kg.get_statistics()
    print(f"\n Graph Statistics:")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Topics: {stats['topics']}")
    print(f"   Resources: {stats['resources']}")
    print(f"   Applications: {stats['applications']}")
    
    # Test querying
    print(f"\n Testing queries...")
    
    # Get resources for a topic
    topic = 'quantum_computing'
    resources = kg.get_resources_for_topic(topic)
    print(f"\n   Resources for '{topic}': {len(resources)}")
    for r in resources[:3]:
        print(f"   - {r.get('title', 'Unknown')[:50]}...")
    
    # Get related topics
    related = kg.get_related_topics('machine_learning')
    print(f"\n   Topics related to 'machine_learning': {related}")
    
    # Get applications
    apps = kg.get_applications_for_topic('fintech')
    print(f"\n   Applications for 'fintech': {[a.get('name') for a in apps]}")
    
    print("\n Knowledge graph built successfully!")


