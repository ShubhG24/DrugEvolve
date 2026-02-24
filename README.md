# Drug Repurposing Analysis Tool

An intelligent web application that discovers potential drug repurposing opportunities by analyzing scientific literature and biomedical knowledge graphs.

<img width="1500" alt="image" src="https://github.com/user-attachments/assets/dd1ffd2b-4837-46e9-97ff-bc36c0d4272b" />


## Features

- **Advanced Semantic Search**: AI-powered matching for diseases, drugs, and genes using sentence transformers
- **Configurable Similarity Threshold**: User-controlled slider to adjust semantic matching strictness (0.0-1.0)
- **Multi-Modal Analysis**: LLM Only, Knowledge Graph Only, or Hybrid LLM + KG approaches
- **Intelligent Matching**: Understands medical synonyms and terminology variations with similarity scores
- **Literature Mining**: Automatically searches and analyzes PubMed abstracts
- **Gene-Disease Mapping**: Identifies genes associated with diseases and drugs targeting those genes
- **FDA Drug Validation**: Cross-references results with FDA-approved drugs
- **Export Functionality**: Download analysis results as CSV files

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Valid email address for PubMed API access

### Installation

1. **Setup environment**
   ```bash
   python3 -m venv dev-env
   source dev-env/bin/activate        # Windows: dev-env\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download knowledge graph data**
   
   Download the kg.csv file (~900MB) from:
   ```
   https://dataverse.harvard.edu/api/access/datafile/6180620
   ```
   Move the file to the main project directory.(Rename it kg.csv)

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Launch**: Navigate to `http://localhost:8501` (usually opens automatically)

2. **Configure settings** in the sidebar:
   - Enter your email address for PubMed API access
   - Provide your Google Gemini API key

3. **Input diseases**:
   - Primary Disease (e.g., "Heart Disease")
   - Comorbidity (e.g., "Diabetes Mellitus")

4. **Configure Semantic Search**:
   - Adjust similarity threshold (0.7-0.9 recommended)
   - Toggle semantic search on/off

5. **Set parameters**: Choose articles count (5-1000) and analysis mode

6. **Run analysis**: Click "Search & Analyze"

### Analysis Modes

- **LLM Only**: Uses Google Gemini AI to analyze PubMed literature for novel insights
- **Knowledge Graph Only**: Uses biomedical knowledge graphs with semantic matching for systematic analysis  
- **Hybrid LLM + KG**: Combines both approaches for comprehensive analysis with cross-validation

### Semantic Search Benefits

- **Better Disease Matching**: Finds related conditions even with different wording
- **Medical Terminology Understanding**: Recognizes synonyms and medical abbreviations  
- **Transparency**: Shows matching results with similarity scores
- **Flexibility**: User-controlled threshold for matching sensitivity

## Project Structure

```
.
├── main.py              # Main Streamlit application
├── cando_tutorial.py    # CANDO platform tutorial script
├── requirements.txt     # Python dependencies
├── products.txt         # FDA approved drugs database
├── kg.csv               # Knowledge graph data (needs to be downloaded manually)
└── README.md            # This file
```

## Configuration

### API Keys
The application requires a Google Gemini API key for LLM analysis. Enter the API key in the sidebar.

### Semantic Search Model
The application automatically downloads the `all-MiniLM-L6-v2` sentence transformer model (~80MB) on first use.

### Data Files
- **Knowledge Graph**: Download `kg.csv` file manually using the link in installation section
- **FDA Drugs**: The `products.txt` file should be in the root directory
- **CANDO Data**: Tutorial data automatically downloaded on first run

## Troubleshooting

### Semantic Search Issues
- If model fails to load, application falls back to fuzzy matching
- First run slower due to model download (~80MB) 
- Requires additional ~200MB RAM

### General Issues
- Ensure `kg.csv` is in root directory (~900MB file size)
- Restart application after entering new Gemini API key
- Check API quota in Google AI Studio


## CANDO Platform Setup
*Note: This section is optional. If you wish to explore additional drug repurposing capabilities using the CANDO platform, follow the setup instructions below.*

### About CANDO
CANDO (Computational Analysis of Novel Drug Opportunities) is a platform for drug repurposing analysis that uses drug-protein interaction signatures to identify potential therapeutic applications for existing drugs.

### Prerequisites for CANDO
- Python 3.8 or higher
- Separate virtual environment recommended

### CANDO Installation

1. **Create CANDO environment**
   ```bash
   python3 -m venv cando-env
   source cando-env/bin/activate      # Windows: cando-env\Scripts\activate
   pip install --upgrade pip
   ```

2. **Install CANDO library**
   ```bash
   pip install cando.py
   ```

3. **Run CANDO tutorial**
   ```bash
   python3 cando_tutorial.py
   ```

### CANDO Features
- **Drug-Protein Interaction Analysis**: Analyzes binding signatures between drugs and protein targets
- **Similarity-Based Predictions**: Uses cosine distance metrics to find similar drugs
- **Disease-Drug Associations**: Maps existing drug-indication relationships
- **Repurposing Predictions**: Identifies new therapeutic applications for existing drugs

### CANDO Tutorial
The `cando_tutorial.py` script demonstrates:
- Data download and preparation
- CANDO object initialization
- Drug-drug similarity computation
- HIV case study analysis
- Compound prediction for specific indications

For more detailed tutorials, refer to the [CANDO Jupyter Notebook](https://github.com/ram-compbio/CANDO/blob/master/CANDO_tutorial.ipynb).

### Switching Between Environments
```bash
# For main drug repurposing tool
source dev-env/bin/activate
streamlit run main.py

# For CANDO analysis
source cando-env/bin/activate
python3 cando_tutorial.py
```

### CANDO Issues
- Ensure no filename conflicts (avoid naming scripts `cando.py`)
- Use separate virtual environment for CANDO
- Tutorial data downloads automatically on first run
