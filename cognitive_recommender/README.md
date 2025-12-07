# Brain Sparks

## Personalized Educational Recommender for Uganda

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Brain Sparks** is a cognitive educational recommender system designed specifically for Ugandan learners. It uses the four pillars of cognitive computing—**Understand**, **Reason**, **Learn**, and **Interact**—to provide personalized learning paths based on natural language queries.

![Brain Sparks Demo](docs/demo.gif)

---

## Features

- **Understand**: Parse natural language queries to extract topic, intent, and context
- **Reason**: Navigate a knowledge graph of 55+ educational resources
- **Learn**: Improve recommendations through user feedback
- **Interact**: Beautiful, responsive web interface
- **Uganda Focus**: Every topic includes Uganda-specific applications

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cognitive_recommender.git
cd cognitive_recommender

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

---

## User Manual

### How to Use Brain Sparks

1. **Enter Your Query**
   - Type any educational topic you want to learn
   - Include context like "for Uganda" or "beginner level"
   - Example: *"Explain the basics of quantum computing and how it could help Uganda"*

2. **View Your Results**
   - **Topic Understanding**: See what the system extracted from your query
   - **Topic Explanation**: Read a summary of the topic
   - **Uganda Relevance**: Discover local applications
   - **Learning Path**: Follow the 3-step recommended sequence

3. **Follow the Learning Path**
   - **Step 1**: Start with foundational content
   - **Step 2**: Build deeper understanding
   - **Step 3**: Test yourself with quizzes

4. **Provide Feedback**
   - Rate the recommendations (1-5 stars)
   - Indicate if they were helpful
   - Your feedback improves future recommendations!

### Sample Queries to Try

```
"Explain machine learning for agriculture in Uganda"
"I want to learn cybersecurity to protect mobile money"
"What is blockchain and how can it help African countries?"
"Teach me web development for building Ugandan businesses"
"How can AI improve healthcare in rural Uganda?"
```

---

## Project Structure

```
cognitive_recommender/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore patterns
│
├── data/                      # Data files
│   ├── educational_content.json  # 55+ educational resources
│   └── feedback.json          # User feedback storage
│
├── docs/                      # Documentation
│   ├── problem_analysis.md    # Scenario analysis
│   ├── implementation_plan.md # Project plan & WBS
│   ├── evaluation_report.md   # Performance evaluation
│   ├── ethical_analysis.md    # Ethics assessment
│   ├── final_report.md        # Comprehensive report
│   └── presentation.md        # Presentation slides
│
├── notebooks/                 # Jupyter notebooks
│   ├── data_pipeline.ipynb    # Data exploration
│   └── understanding_reasoning.ipynb  # NLP & KG demo
│
└── src/                       # Source code
    ├── app.py                 # Streamlit application
    ├── nlp_utils.py           # NLP processing
    ├── kg_utils.py            # Knowledge graph
    └── recommender.py         # Recommendation engine
```

---

## 🔧 Technical Details

### Architecture

```
┌─────────────────────────────────────────────┐
│              Brain Sparks                    │
├─────────────────────────────────────────────┤
│  INTERACT     UNDERSTAND     REASON         │
│  (Streamlit)  (NLTK)        (NetworkX)      │
│       ↕            ↕             ↕          │
│              LEARN (JSON)                   │
│                   ↕                         │
│              DATA (JSON)                    │
└─────────────────────────────────────────────┘
```

### Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web UI | Streamlit | Interactive interface |
| NLP | NLTK | Query parsing |
| Knowledge Graph | NetworkX | Content relationships |
| Similarity | Scikit-learn | Content matching |
| Storage | JSON | Data persistence |

### The Four Pillars

1. **UNDERSTAND** (`nlp_utils.py`)
   - Query tokenization and preprocessing
   - Topic identification using keyword matching
   - Uganda context detection
   - Intent classification

2. **REASON** (`kg_utils.py`)
   - Knowledge graph construction
   - Topic-resource relationships
   - Learning path generation
   - Related topic discovery

3. **LEARN** (`recommender.py`)
   - Feedback collection
   - Score adjustment based on ratings
   - Continuous improvement

4. **INTERACT** (`app.py`)
   - Natural language input
   - Visual learning paths
   - Interactive quizzes
   - Feedback forms

---

## Evaluation

### Performance Metrics

| Metric | Result |
|--------|--------|
| Topic Accuracy | 85% |
| Uganda Context Detection | 90% |
| Precision@3 | 73% |
| Response Time | <0.5s |

### vs. Keyword Baseline

Brain Sparks outperforms simple keyword search by **62%** in recommendation precision.

---

## Ethical Considerations

### Privacy
- No personal data collected
- Anonymous feedback only
- All data stored locally

### Fairness
- Balanced topic coverage
- Diverse source inclusion
- No demographic profiling

### Transparency
- Confidence scores displayed
- Recommendations explained
- Limitations acknowledged

---

## Roadmap

### Current Version (v1.0)
- [x] Core cognitive system
- [x] 55 educational resources
- [x] Uganda context integration
- [x] Feedback collection

### Future Plans
- [ ] Expand to 200+ resources
- [ ] Local language support (Luganda)
- [ ] Mobile application
- [ ] Multi-turn conversations
- [ ] User profiles

---

## Author

**Rugogamu Noela**  
Uganda Christian University (UCU)  
Cognitive Computing Project  
December 2025

---

## License

This project is for academic purposes. Content is curated from open-source educational materials.

---

## Acknowledgments

- Uganda Christian University
- Open-source educational platforms
- NLTK, NetworkX, Streamlit communities

---

## Support

For questions or issues:
1. Check the documentation in `/docs`
2. Review the notebooks in `/notebooks`
3. Open an issue on GitHub

---

*"Making technology education relevant for Uganda"*

**© 2025 Brain Sparks - Built by Rugogamu Noela, Uganda Christian University (UCU)**


