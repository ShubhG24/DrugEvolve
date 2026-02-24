import streamlit as st
import requests
from Bio import Entrez, Medline
import csv
import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import pandas as pd
import difflib
from collections import defaultdict
import ssl
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="DrugEvolve",
    page_icon="🔄",
    layout="wide",
)

# Load environment variables
load_dotenv()

@st.cache_resource
def configure_entrez(email):
    Entrez.email = email
    # Configure SSL context to handle certificate issues
    try:
        # Create an SSL context that doesn't verify certificates (for development/testing)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Apply the SSL context to urllib
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
    except Exception as e:
        st.warning(f"SSL configuration warning: {e}")

@st.cache_resource
def configure_gemini():
    api_key = st.session_state.get('api_key', '')
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    return None

@st.cache_data
def load_kg():
    kg_df = pd.read_csv("kg.csv", low_memory=False)
    kg_df['x_type'] = kg_df['x_type'].str.lower()
    kg_df['y_type'] = kg_df['y_type'].str.lower()
    kg_df['relation'] = kg_df['relation'].str.lower()
    kg_df['display_relation'] = kg_df['display_relation'].str.lower()
    return kg_df

@st.cache_resource
def load_semantic_model():
    """Load the sentence transformer model for semantic search"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading semantic model: {e}")
        return None

def semantic_search(query, candidates, model, threshold=0.7):
    """
    Perform semantic search to find the best matching candidate
    
    Args:
        query (str): The search term
        candidates (list): List of candidate terms to search through
        model: SentenceTransformer model
        threshold (float): Minimum similarity score (0-1)
    
    Returns:
        tuple: (best_match, similarity_score) or (None, 0) if no match above threshold
    """
    if not model or not candidates:
        return None, 0
    
    try:
        # Encode query and candidates
        query_embedding = model.encode([query])
        candidate_embeddings = model.encode(candidates)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return candidates[best_idx], best_score
        else:
            return None, best_score
            
    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return None, 0

def get_closest_disease_name_semantic(disease_name, kg_df, model, threshold=0.7):
    """Find closest disease name using semantic search"""
    disease_names = kg_df[kg_df['x_type'] == 'disease']['x_name'].dropna().unique().tolist()
    best_match, score = semantic_search(disease_name, disease_names, model, threshold)
    return best_match, score

def get_closest_drug_name_semantic(drug_name, kg_df, model, threshold=0.7):
    """Find closest drug name using semantic search"""
    drug_names = kg_df[kg_df['x_type'] == 'drug']['x_name'].dropna().unique().tolist()
    best_match, score = semantic_search(drug_name, drug_names, model, threshold)
    return best_match, score

def get_closest_gene_name_semantic(gene_name, kg_df, model, threshold=0.7):
    """Find closest gene name using semantic search"""
    gene_names = kg_df[kg_df['y_type'] == 'gene/protein']['y_name'].dropna().unique().tolist()
    best_match, score = semantic_search(gene_name, gene_names, model, threshold)
    return best_match, score

# Fuzzy match helper for KG (keeping as backup)
def get_closest_disease_name_kg(disease_name, kg_df, min_similarity=0.8):
    disease_name_lower = disease_name.lower()
    disease_names = kg_df[kg_df['x_type'] == 'disease']['x_name'].dropna().str.lower().unique()
    
    # Manual best match using similarity ratio
    best_match = None
    best_score = 0

    for candidate in disease_names:
        score = difflib.SequenceMatcher(None, disease_name_lower, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= min_similarity:
        return best_match
    else:
        return None

# Gene retrieval for KG
def get_genes_for_disease_kg(disease_name, kg_df, model=None, threshold=0.7, use_semantic=True):
    if use_semantic and model:
        matched_name, score = get_closest_disease_name_semantic(disease_name, kg_df, model, threshold)
    else:
        matched_name = get_closest_disease_name_kg(disease_name, kg_df)
        score = None
    
    if not matched_name:
        return [], score
    
    # Use exact matching for the matched disease name
    genes = kg_df[
        (kg_df['x_name'] == matched_name) &
        (kg_df['x_type'] == 'disease') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin({'associated with', 'expression present', 'expression absent'}))
    ]['y_name'].dropna().unique().tolist()
    
    return genes, score

# Drug retrieval targeting genes for KG
def get_drugs_targeting_genes_kg(genes, kg_df):
    return kg_df[
        (kg_df['y_name'].isin(genes)) &
        (kg_df['x_type'] == 'drug') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin({'target', 'enzyme', 'carrier', 'transporter', 'ppi'}))
    ][['x_name', 'y_name']].drop_duplicates()

# Count genes targeted by each drug for KG
def count_genes_targeted_kg(drug_targets):
    drug_gene_counts = defaultdict(set)
    for _, row in drug_targets.iterrows():
        drug = row['x_name']
        gene = row['y_name']
        drug_gene_counts[drug].add(gene)
    return {drug: len(genes) for drug, genes in drug_gene_counts.items()}

# Rank drugs by the number of genes targeted for KG
def rank_drugs_by_gene_count_kg(drug_gene_counts):
    return sorted(drug_gene_counts.items(), key=lambda item: item[1], reverse=True)

# Function to visualize drug targets
def visualize_drug_targets(ranked_drugs, drug_targets, disease_name):
    if not ranked_drugs:
        st.info(f"No drugs found targeting genes associated with {disease_name}.")
        return

    st.subheader(f"Top Drugs Targeting Genes Associated with {disease_name} (KG Analysis)")
    for drug, count in ranked_drugs[:10]:
        targeted_genes = drug_targets[drug_targets['x_name'] == drug]['y_name'].unique().tolist()
        st.markdown(f"**{drug}**: Targets **{count}** genes - {', '.join(targeted_genes)}")

# Build PubMed query
def build_pubmed_query(primary_disease, comorbidity):
    primary_query = f'"{primary_disease}"[MeSH Terms] OR "{primary_disease}"[Title/Abstract]'
    comorbidity_query = f'"{comorbidity}"[MeSH Terms] OR "{comorbidity}"[Title/Abstract]'
    return f"(({primary_query}) AND ({comorbidity_query})) AND (\"drug therapy\"[Subheading] OR \"pharmacology\"[Subheading] OR \"therapeutic use\"[Subheading] OR \"treatment\"[Title/Abstract]) AND humans[Filter]"

# Fetch PubMed abstracts
def fetch_pubmed_abstracts(query, max_results=25):
    try:
        with st.spinner('Searching PubMed...'):
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(search_handle)
            pmids = record["IdList"]

        if not pmids:
            return []

        abstracts = []
        with st.spinner('Fetching abstracts from PubMed...'):
            # Progress bar
            progress_bar = st.progress(0)
            chunk_size = max(1, len(pmids) // 10)  # divide into 10 chunks

            for i in range(0, len(pmids), chunk_size):
                batch_ids = pmids[i:i + chunk_size]
                fetch_handle = Entrez.efetch(db="pubmed", id=",".join(batch_ids), rettype="medline", retmode="text")
                batch_records = list(Medline.parse(fetch_handle))
                abstracts.extend(batch_records)
                progress_bar.progress(min(1.0, (i + chunk_size) / len(pmids)))

            progress_bar.empty()
            return abstracts
    except Exception as e:
        st.error(f"Error fetching PubMed data: {str(e)}")
        return []

# Use Gemini to extract relationships
def extract_relationships(abstracts, primary_disease, comorbidity):
    model = configure_gemini()
    if not model:
        st.error("Gemini API key not configured properly")
        return None
    full_text = "\n".join([
        f"Title: {record.get('TI', '')}\nAbstract: {record.get('AB', '')}\n"
        for record in abstracts if record.get('AB')
    ])
    prompt = f"""Analyze the following collection of medical abstracts and extract relationships between diseases,
    comorbidities, and drugs. Only use the abstracts provided.

    **Your Task**:
    - Identify the **primary disease**: {primary_disease}.
    - Identify the **comorbidity** being analyzed: {comorbidity}.
    - List **drugs used to treat the primary disease**.
    - List **drugs used to treat the comorbidity**.
    - Identify **drugs that are effective for both conditions**.\n    - For each drug that treats both conditions, provide:
      1. A score from 1-10 on potential for repurposing (10 being highest)(If the drug is not explicitly mentioned in the literature, but is biologically plausible based on the knowledge graph, assign a score between 3–6 based on inferred mechanism.)
      2. Evidence from literature for its dual efficacy
      3. Mechanism of action explaining why it might work for both conditions
      4. Potential molecular targets relevant to both conditions (if mentioned)

    **Output Format (JSON)**:
    {{
        "primary_disease": "{primary_disease}",
        "comorbidity": "{comorbidity}",
        "drugs_primary_disease": ["Drug1", "Drug2"],
        "drugs_comorbidity": ["DrugA", "DrugB"],
        "shared_treatments": [
            {{
                "drug": "Drug Name",
                "repurposing_score": 8,
                "primary_disease_treatment": true,
                "comorbidity_treatment": true,
                "evidence": "Brief evidence from literature",
                "mechanism_of_action": "Inhibits X receptor, reducing Y",
                "molecular_targets": ["target1", "target2"]
            }}
        ],
        "explanation": "Overall explanation for drugs that are effective for both conditions."
    }}

    **Medical Abstracts**:
    {full_text}
    """
    try:
        with st.spinner('Analyzing abstracts with AI...'):
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
            text_response = re.sub(r'```json\s*', '', response.text)
            text_response = re.sub(r'```\s*', '', text_response).strip()
            return json.loads(text_response)
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

# Function to save data to CSV
def save_to_csv(data, primary_disease, comorbidity):
    if not data or 'shared_treatments' not in data:
        return None
    filename = f"{primary_disease}_{comorbidity}_repurposing_results.csv"
    if data['shared_treatments']:
        df = pd.DataFrame(data['shared_treatments'])
        df.to_csv(filename, index=False)
        return filename
    return None

def load_fda_approved_drugs(txt_path):
    approved_drugs = set()
    with open(txt_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            fields = line.strip().split('~')
            if len(fields) >= 1:
                drug = fields[0].strip().upper()
                if drug:
                    approved_drugs.add(drug)
    return approved_drugs


# Function to visualize drug rankings
def visualize_drug_rankings(data, approved_drug_names=None):
    if not data or 'shared_treatments' not in data or not data['shared_treatments']:
        st.info("No shared treatments found to visualize.")
        return

    shared_treatments = data['shared_treatments']
    sorted_drugs = sorted(shared_treatments, key=lambda x: x.get('repurposing_score', 0), reverse=True)
    df = pd.DataFrame(sorted_drugs)
    display_df = df[['drug', 'repurposing_score', 'mechanism_of_action', 'evidence', 'molecular_targets']].copy()
    display_df.columns = ['Drug', 'Repurposing Score', 'Mechanism of Action', 'Evidence', 'Potential Molecular Targets']
    st.subheader("Detailed Drug Information")
    st.dataframe(display_df)

    if approved_drug_names:
        approved = []
        for drug in display_df['Drug']:
            drug_upper = str(drug).upper()
            for approved_name in approved_drug_names:
                if approved_name in drug_upper or drug_upper in approved_name:
                    approved.append(drug)
                    break

        if approved:
            st.subheader("FDA Approved Drugs:")
            st.write("Others that are not shown might be drug groups/experimental/investigational/approved in other countries")
            for drug in approved:
                st.markdown(f"- {drug}")
        else:
            st.info("No FDA approved drugs found among the results.")

# Function to score drugs comprehensively for KG analysis
def score_drugs_kg(drug_targets, shared_genes, genes_primary, genes_comorbidity):
    """
    Score drugs based on multiple factors:
    1. Number of shared genes targeted
    2. Number of primary disease genes targeted
    3. Number of comorbidity genes targeted
    4. Overall target specificity
    """
    drug_scores = {}
    drug_gene_details = {}
    
    for drug in drug_targets['x_name'].unique():
        drug_data = drug_targets[drug_targets['x_name'] == drug]
        targeted_genes = set(drug_data['y_name'].unique())
        
        # Calculate component scores
        shared_targets = len(targeted_genes.intersection(set(shared_genes)))
        primary_targets = len(targeted_genes.intersection(set(genes_primary)))
        comorbidity_targets = len(targeted_genes.intersection(set(genes_comorbidity)))
        total_targets = len(targeted_genes)
        
        # Weighted scoring system
        shared_score = shared_targets * 3.0  # Highest weight for shared genes
        primary_score = primary_targets * 1.5
        comorbidity_score = comorbidity_targets * 1.5
        specificity_bonus = min(2.0, 10.0 / max(1, total_targets))  # Bonus for specificity
        
        final_score = shared_score + primary_score + comorbidity_score + specificity_bonus
        
        drug_scores[drug] = final_score
        drug_gene_details[drug] = {
            'total_targets': total_targets,
            'shared_targets': shared_targets,
            'primary_targets': primary_targets,
            'comorbidity_targets': comorbidity_targets,
            'targeted_genes': list(targeted_genes),
            'score': final_score
        }
    
    return drug_scores, drug_gene_details

# Function to group drugs by target gene count
def group_drugs_by_gene_count(drug_gene_details):
    """Group drugs by the number of genes they target."""
    grouped = {}
    for drug, details in drug_gene_details.items():
        count = details['total_targets']
        if count not in grouped:
            grouped[count] = []
        grouped[count].append((drug, details))
    
    # Sort groups by gene count (descending) and drugs within groups by score
    sorted_groups = {}
    for count in sorted(grouped.keys(), reverse=True):
        sorted_groups[count] = sorted(grouped[count], key=lambda x: x[1]['score'], reverse=True)
    
    return sorted_groups

# Function to display drug details in an expandable format
def display_drug_details(drug, details, genes_primary, genes_comorbidity, shared_genes):
    """Display detailed information about a drug's targets and scoring."""
    # Use container instead of expander to avoid nesting issues
    with st.container():
        st.write(f"**🔍 {drug}** (Score: {details['score']:.2f})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Targets", details['total_targets'])
        with col2:
            st.metric("Shared Gene Targets", details['shared_targets'])
        with col3:
            st.metric("Primary Disease Targets", details['primary_targets'])
        with col4:
            st.metric("Comorbidity Targets", details['comorbidity_targets'])
        
        # Show targeted genes by category
        if details['shared_targets'] > 0:
            shared_targeted = [g for g in details['targeted_genes'] if g in shared_genes]
            st.write(f"**Shared Genes Targeted:** {', '.join(shared_targeted)}")
        
        if details['primary_targets'] > 0:
            primary_targeted = [g for g in details['targeted_genes'] if g in genes_primary and g not in shared_genes]
            if primary_targeted:
                st.write(f"**Primary Disease Genes:** {', '.join(primary_targeted)}")
        
        if details['comorbidity_targets'] > 0:
            comorbidity_targeted = [g for g in details['targeted_genes'] if g in genes_comorbidity and g not in shared_genes]
            if comorbidity_targeted:
                st.write(f"**Comorbidity Genes:** {', '.join(comorbidity_targeted)}")
        
        st.divider()

def get_semantic_matches_summary(primary_disease, comorbidity, kg_df, model, threshold):
    """Get semantic matching results and scores for both diseases"""
    primary_match, primary_score = get_closest_disease_name_semantic(primary_disease, kg_df, model, threshold)
    comorbidity_match, comorbidity_score = get_closest_disease_name_semantic(comorbidity, kg_df, model, threshold)
    
    return {
        'primary': {'match': primary_match, 'score': primary_score, 'original': primary_disease},
        'comorbidity': {'match': comorbidity_match, 'score': comorbidity_score, 'original': comorbidity}
    }

def display_semantic_matching_results(matching_results):
    """Display the semantic matching results in the UI"""
    st.subheader("🔍 Semantic Matching Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Disease:**")
        primary = matching_results['primary']
        if primary['match']:
            st.success(f"✅ **{primary['original']}** → **{primary['match']}**")
            st.write(f"Similarity Score: {primary['score']:.3f}")
        else:
            st.error(f"❌ No match found for **{primary['original']}**")
            st.write(f"Best Score: {primary['score']:.3f}")
    
    with col2:
        st.write("**Comorbidity:**")
        comorbidity = matching_results['comorbidity']
        if comorbidity['match']:
            st.success(f"✅ **{comorbidity['original']}** → **{comorbidity['match']}**")
            st.write(f"Similarity Score: {comorbidity['score']:.3f}")
        else:
            st.error(f"❌ No match found for **{comorbidity['original']}**")
            st.write(f"Best Score: {comorbidity['score']:.3f}")
    
    st.divider()

def main():
    # App header with logo and title
    col1, col2 = st.columns([0.5, 6])
    with col1:
        st.markdown("# 🔄")
    with col2:
        st.title("DrugEvolve")
        st.markdown("*Evolving Drug Discovery Through AI & Knowledge Graphs*")
    
    st.write("Discover potential drug repurposing opportunities by analyzing literature and knowledge graphs.")
    FDA_TXT_PATH = "products.txt"  # replace with your file name
    approved_drug_names = load_fda_approved_drugs(FDA_TXT_PATH)

    with st.sidebar:
        st.header("Settings")
        user_email = st.text_input("Email for PubMed API", value="your_email@example.com")
        configure_entrez(user_email)
        
        st.subheader("Google Gemini API Configuration")
        
        # API Key Configuration
        with st.form("api_key_form"):
            api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Gemini API key to enable AI analysis")
            submitted = st.form_submit_button("Enter", type="primary")
            if submitted:
                if api_key:
                    st.session_state['api_key'] = api_key
                    st.success("API key saved!")
                else:
                    st.error("Please enter an API key")
        
        # Display current API key status
        if st.session_state.get('api_key'):
            st.success("🔑 API Key is configured")
        else:
            st.warning("🔑 API Key not set - required for LLM analysis")

        # Add helpful description
        with st.expander("ℹ️ API Key Help"):
            st.write("""
            **Getting your Gemini API Key:**
            1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Create a new API key
            3. Copy and paste it above
            4. Click 'Enter'
            
            **Troubleshooting:**
            - If you get Api key error, please restart the app from terminal
            - Make sure your API key has proper permissions
            - Check that you haven't exceeded your API quota
            """)

    col1, col2 = st.columns(2)
    with col1:
        primary_disease = st.text_input("Primary Disease", value="Heart Disease")
    with col2:
        comorbidity = st.text_input("Comorbidity", value="Diabetes Mellitus")

    # Semantic Search Settings
    st.subheader("🔍 Semantic Search Settings")
    
    with st.expander("ℹ️ About Semantic Search", expanded=False):
        st.write("""
        **Semantic Search** uses AI to understand the meaning and context of medical terms, 
        providing much better matching than simple text comparison.
        
        **Benefits:**
        - Understands medical synonyms (e.g., "Heart Disease" ↔ "Cardiovascular Disease")
        - Handles different terminology (e.g., "Diabetes" ↔ "Diabetes Mellitus")
        - Finds related conditions even with different wording
        
        **Threshold Guide:**
        - **0.9-1.0**: Very strict matching (almost exact)
        - **0.7-0.9**: Good balance (recommended)
        - **0.5-0.7**: More flexible matching
        - **0.3-0.5**: Very loose matching (may include unrelated terms)
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        semantic_threshold = st.slider(
            "Semantic Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Higher values require more exact matches. Lower values allow more flexible matching."
        )
    
    with col2:
        use_semantic_search = st.checkbox(
            "Use Semantic Search", 
            value=True,
            help="Use AI-powered semantic search instead of fuzzy string matching"
        )

    max_results = st.slider("Maximum number of articles to analyze", 5, 1000, 50, step=25)
    analysis_mode = st.radio("Analysis Mode (API Key Required for LLM/LLM+KG)", ["LLM Only", "KG Only", "LLM + KG"], index=0)


    if st.button("Search & Analyze", type="primary"):
        if not primary_disease or not comorbidity:
            st.warning("Please enter both primary disease and comorbidity.")
            return
        
        if analysis_mode == "LLM Only":
            query = build_pubmed_query(primary_disease, comorbidity)
            abstracts = fetch_pubmed_abstracts(query, max_results)

            if not abstracts:
                st.warning(f"No abstracts found for {primary_disease} and {comorbidity}.")
                return

            st.success(f"Found {len(abstracts)} relevant articles.")
            results = extract_relationships(abstracts, primary_disease, comorbidity)
            
            if results:
                st.header(f"Analysis Results (LLM Only): {primary_disease} + {comorbidity}")
                st.subheader("Shared Treatments")
                shared_count = len(results.get('shared_treatments', []))
                if shared_count > 0:
                    st.success(f"Found {shared_count} potential drugs for repurposing!")
                    visualize_drug_rankings(results, approved_drug_names) 
                    csv_file = save_to_csv(results, primary_disease, comorbidity)
                    if csv_file:
                        with open(csv_file, 'rb') as f:
                            st.download_button(
                                label="Download Results as CSV",
                                data=f,
                                file_name=csv_file,
                                mime="text/csv"
                            )
                else:
                    st.info("No shared treatments found between the diseases based on the analyzed abstracts.")

                if 'explanation' in results:
                    st.subheader("General Analysis")
                    st.write(results['explanation'])

                st.subheader("Raw JSON Output")
                st.json(results)

        elif analysis_mode == "KG Only":
            kg_df = load_kg()
            
            # Load semantic model if using semantic search
            model = None
            if use_semantic_search:
                model = load_semantic_model()
                if not model:
                    st.error("Failed to load semantic search model. Falling back to fuzzy matching.")
                    use_semantic_search = False

            # Get semantic matching results
            if use_semantic_search and model:
                matching_results = get_semantic_matches_summary(primary_disease, comorbidity, kg_df, model, semantic_threshold)
                display_semantic_matching_results(matching_results)
                
                matched_primary_disease = matching_results['primary']['match']
                matched_comorbidity = matching_results['comorbidity']['match']
            else:
                matched_primary_disease = get_closest_disease_name_kg(primary_disease, kg_df)
                matched_comorbidity = get_closest_disease_name_kg(comorbidity, kg_df)
                st.info(f"Fuzzy Matched Primary Disease: {matched_primary_disease}")
                st.info(f"Fuzzy Matched Comorbidity: {matched_comorbidity}")

            if matched_primary_disease and matched_comorbidity:
                genes_primary, _ = get_genes_for_disease_kg(
                    matched_primary_disease, kg_df, model, semantic_threshold, use_semantic_search
                )
                genes_comorbidity, _ = get_genes_for_disease_kg(
                    matched_comorbidity, kg_df, model, semantic_threshold, use_semantic_search
                )
                shared_genes = list(set(genes_primary) & set(genes_comorbidity))

                st.info(f"Genes for Primary Disease: {', '.join(genes_primary) if genes_primary else 'No genes found.'}")
                st.info(f"Genes for Comorbidity: {', '.join(genes_comorbidity) if genes_comorbidity else 'No genes found.'}")
                st.info(f"Shared Genes: {', '.join(shared_genes) if shared_genes else 'No shared genes found.'}")

                st.header(f"Drug Repurposing Analysis (KG Only): {primary_disease} + {comorbidity}")

                # Get all drugs targeting any relevant genes
                all_relevant_genes = list(set(genes_primary + genes_comorbidity))
                
                if all_relevant_genes:
                    # Get all drug targets for relevant genes
                    drug_targets_all = get_drugs_targeting_genes_kg(all_relevant_genes, kg_df)
                    # Fix regex pattern by escaping parentheses
                    drug_targets_all = drug_targets_all[~drug_targets_all['x_name'].str.contains(r'\(fibroblast|keratinocyte|neonatal|ovine|recombinant\)', case=False)]
                    
                    if len(drug_targets_all) > 0:
                        # Score all drugs comprehensively
                        drug_scores, drug_gene_details = score_drugs_kg(
                            drug_targets_all, shared_genes, genes_primary, genes_comorbidity
                        )
                        
                        # Sort drugs by score
                        ranked_drugs_by_score = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display Top 10 drugs
                        st.subheader("🏆 Top 10 Drug Candidates (Comprehensive Scoring)")
                        st.write("*Scoring considers shared gene targets (3x weight), disease-specific targets (1.5x weight), and target specificity bonus.*")
                        
                        # Create a single expander for the top 10 drugs
                        with st.expander("View Top 10 Drug Details", expanded=True):
                            for i, (drug, score) in enumerate(ranked_drugs_by_score[:10], 1):
                                details = drug_gene_details[drug]
                                
                                # Create a nice display for each top drug
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{i}. {drug}**")
                                    target_summary = f"Targets {details['total_targets']} genes"
                                    if details['shared_targets'] > 0:
                                        target_summary += f" ({details['shared_targets']} shared)"
                                    st.write(target_summary)
                                with col2:
                                    st.metric("Score", f"{score:.2f}")
                                
                                # Show details using container instead of nested expander
                                display_drug_details(drug, details, genes_primary, genes_comorbidity, shared_genes)
                        
                        # Group drugs by target gene count for dropdown
                        st.subheader("📊 All Drugs Grouped by Number of Target Genes")
                        grouped_drugs = group_drugs_by_gene_count(drug_gene_details)
                        
                        # Create dropdown for each gene count group with simple drug lists
                        for gene_count in sorted(grouped_drugs.keys(), reverse=True):
                            drugs_in_group = grouped_drugs[gene_count]
                            group_label = f"Drugs targeting {gene_count} gene{'s' if gene_count != 1 else ''} ({len(drugs_in_group)} drugs)"
                            
                            with st.expander(group_label):
                                # Simple list of drugs with their scores
                                for drug, details in drugs_in_group:
                                    st.write(f"• **{drug}** (Score: {details['score']:.2f})")
                    
                    else:
                        st.warning("No drugs found targeting the identified genes after filtering.")
                
                else:
                    st.warning("No genes found associated with either the primary disease or the comorbidity.")

            else:
                st.warning("Could not find exact or close matches for both diseases in the Knowledge Graph.")

        elif analysis_mode == "LLM + KG":
            st.info("LLM + KG Integration Analysis")
            kg_df = load_kg()
            
            # Load semantic model if using semantic search
            model = None
            if use_semantic_search:
                model = load_semantic_model()
                if not model:
                    st.error("Failed to load semantic search model. Falling back to fuzzy matching.")
                    use_semantic_search = False

            # Get semantic matching results
            if use_semantic_search and model:
                matching_results = get_semantic_matches_summary(primary_disease, comorbidity, kg_df, model, semantic_threshold)
                display_semantic_matching_results(matching_results)
                
                matched_primary_disease = matching_results['primary']['match']
                matched_comorbidity = matching_results['comorbidity']['match']
            else:
                matched_primary_disease = get_closest_disease_name_kg(primary_disease, kg_df)
                matched_comorbidity = get_closest_disease_name_kg(comorbidity, kg_df)

            if not matched_primary_disease or not matched_comorbidity:
                st.warning("Could not find close matches for both diseases in the Knowledge Graph.")
                return

            genes_primary, _ = get_genes_for_disease_kg(
                matched_primary_disease, kg_df, model, semantic_threshold, use_semantic_search
            )
            genes_comorbidity, _ = get_genes_for_disease_kg(
                matched_comorbidity, kg_df, model, semantic_threshold, use_semantic_search
            )
            shared_genes = list(set(genes_primary) & set(genes_comorbidity))

            # Get drugs targeting genes for both diseases
            all_relevant_genes = list(set(genes_primary + genes_comorbidity))
            drug_targets_all = get_drugs_targeting_genes_kg(all_relevant_genes, kg_df)
            drug_targets_all = drug_targets_all[~drug_targets_all['x_name'].str.contains(r'\(fibroblast|keratinocyte|neonatal|ovine|recombinant\)', case=False)]
            
            if len(drug_targets_all) == 0:
                st.warning("No drug targets found in Knowledge Graph for the identified genes.")
                return
            
            # Score all drugs and get top candidates
            drug_scores, drug_gene_details = score_drugs_kg(
                drug_targets_all, shared_genes, genes_primary, genes_comorbidity
            )
            ranked_drugs_by_score = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)

            # Create simple KG context for LLM
            kg_context = f"Knowledge Graph Drug Candidates for {primary_disease} + {comorbidity}:\n"
            kg_context += f"Genes associated with {primary_disease}: {', '.join(genes_primary[:10])}\n"
            kg_context += f"Genes associated with {comorbidity}: {', '.join(genes_comorbidity[:10])}\n"
            kg_context += f"Shared genes: {', '.join(shared_genes)}\n\n"
            kg_context += "Top drug candidates that target relevant genes:\n"
            
            for i, (drug, score) in enumerate(ranked_drugs_by_score[:15], 1):
                details = drug_gene_details[drug]
                kg_context += f"{i}. {drug} - targets {details['total_targets']} genes"
                if details['shared_targets'] > 0:
                    kg_context += f" ({details['shared_targets']} shared genes)"
                kg_context += f"\n"

            # Fetch abstracts
            query = build_pubmed_query(primary_disease, comorbidity)
            abstracts = fetch_pubmed_abstracts(query, max_results)

            if not abstracts:
                st.warning("No abstracts found for the diseases.")
                return

            model = configure_gemini()
            full_text = "\n".join([
                f"Title: {record.get('TI', '')}\nAbstract: {record.get('AB', '')}\n"
                for record in abstracts if record.get('AB')
            ])

            prompt = f"""Analyze the following medical abstracts and knowledge graph data to extract drug repurposing opportunities.

Primary Disease: {primary_disease}
Comorbidity: {comorbidity}

{kg_context}

**Your Task**:
- Analyze the medical abstracts for drug relationships
- Consider the knowledge graph drug candidates provided above
- Identify drugs that could be effective for both conditions
- Prioritize drugs that appear in literature AND/OR have strong biological rationale from the knowledge graph

For each drug that could treat both conditions, provide:
1. A repurposing score from 1-10 (10 being highest confidence)
2. Evidence from literature and/or biological rationale from gene targets
3. Mechanism of action explaining why it might work for both conditions
4. Molecular targets relevant to both conditions

**Output Format (JSON)**:
{{
    "primary_disease": "{primary_disease}",
    "comorbidity": "{comorbidity}",
    "drugs_primary_disease": ["Drug1", "Drug2"],
    "drugs_comorbidity": ["DrugA", "DrugB"],
    "shared_treatments": [
        {{
            "drug": "Drug Name",
            "repurposing_score": 8,
            "primary_disease_treatment": true,
            "comorbidity_treatment": true,
            "evidence": "Evidence from literature and/or gene target rationale",
            "mechanism_of_action": "Inhibits X receptor, reducing Y",
            "molecular_targets": ["target1", "target2"]
        }}
    ],
    "explanation": "Overall explanation for the drug repurposing opportunities identified."
}}

**Medical Abstracts**:
{full_text}
"""

            try:
                with st.spinner("Analyzing with LLM + KG integration..."):
                    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                    text_response = re.sub(r'```json\s*', '', response.text)
                    text_response = re.sub(r'```\s*', '', text_response).strip()
                    results = json.loads(text_response)

                st.header(f"Analysis Results (LLM + KG): {primary_disease} + {comorbidity}")
                st.subheader("Shared Treatments")
                shared_count = len(results.get('shared_treatments', []))
                if shared_count > 0:
                    st.success(f"Found {shared_count} potential drugs for repurposing!")
                    visualize_drug_rankings(results, approved_drug_names) 
                    csv_file = save_to_csv(results, primary_disease, comorbidity)
                    if csv_file:
                        with open(csv_file, 'rb') as f:
                            st.download_button(
                                label="Download Results as CSV",
                                data=f,
                                file_name=csv_file,
                                mime="text/csv"
                            )
                else:
                    st.info("No shared treatments found between the diseases based on the analyzed data.")

                if 'explanation' in results:
                    st.subheader("General Analysis")
                    st.write(results['explanation'])

                st.subheader("Raw JSON Output")
                st.json(results)

            except Exception as e:
                st.error(f"Error during LLM + KG analysis: {str(e)}")
                st.error("Please check your API key and try again.")


if __name__ == "__main__":
    main()
