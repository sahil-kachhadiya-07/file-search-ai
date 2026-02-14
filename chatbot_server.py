"""
chatbot_server.py - Flask app with NLP & context-aware file search
=================================================================
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

# Load .env from project root (same directory as this file)
load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================================
# CONFIGURATION (from environment variables)
# ============================================================================
app = Flask(__name__)
app.config.from_mapping(
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", ""),
    STORE_CONFIG_FILE=os.getenv("STORE_CONFIG_FILE", "store_config.json"),
    DEBUG=os.getenv("FLASK_DEBUG", "true").lower() in ("1", "true", "yes"),
    HOST=os.getenv("FLASK_HOST", "0.0.0.0"),
    PORT=int(os.getenv("FLASK_PORT", "5000")),
)

# Genai client (requires GEMINI_API_KEY in env or .env)
_api_key = app.config["GEMINI_API_KEY"]
if not _api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Set it in your environment or in a .env file."
    )
client = genai.Client(api_key=_api_key)

# ============================================================================
# ENHANCED FILTER EXTRACTION
# ============================================================================
def load_store_config():
    """Load indexed file store configuration."""
    config_path = app.config.get("STORE_CONFIG_FILE", "store_config.json")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def extract_filters_from_query(query_text, config):
    """
    Enhanced filter extraction with better NLP understanding.
    Handles variations like "client b", "Client B", "clientb", etc.
    """
    query_lower = query_text.lower()
    
    filters = {
        'client': None,
        'year': None,
        'month': None
    }
    
    # ENHANCED CLIENT DETECTION
    if config and 'stats' in config and 'clients' in config['stats']:
        for indexed_client in config['stats']['clients']:
            # Generate all possible variations
            base_name = indexed_client.replace('_', ' ')  # "client_a" → "client a"
            
            variations = [
                indexed_client,              # "client_a"
                base_name,                   # "client a"
                indexed_client.replace('_', ''),  # "clienta"
                base_name.replace(' ', ''),  # "clienta"
                base_name.title(),           # "Client A"
                base_name.upper(),           # "CLIENT A"
            ]
            
            for variation in variations:
                # Use word boundary for exact matching
                pattern = r'\b' + re.escape(variation.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    filters['client'] = indexed_client
                    break
            
            if filters['client']:
                break
    
    # YEAR DETECTION (unchanged)
    if config and 'stats' in config and 'years' in config['stats']:
        for year in config['stats']['years']:
            if year in query_text:
                filters['year'] = year
                break
    
    # MONTH DETECTION (unchanged)
    month_patterns = {
        'january': ['january', 'jan'],
        'february': ['february', 'feb'],
        'march': ['march', 'mar'],
        'april': ['april', 'apr'],
        'may': ['may'],
        'june': ['june', 'jun'],
        'july': ['july', 'jul'],
        'august': ['august', 'aug'],
        'september': ['september', 'sep', 'sept'],
        'october': ['october', 'oct'],
        'november': ['november', 'nov'],
        'december': ['december', 'dec'],
        'others': ['others', 'other', 'misc']
    }
    
    for full_month, variants in month_patterns.items():
        for variant in variants:
            if re.search(r'\b' + variant + r'\b', query_lower):
                filters['month'] = full_month
                break
        if filters['month']:
            break
    
    return filters

def build_search_filter(filters):
    """Build metadata filter string."""
    conditions = []
    
    if filters['client']:
        conditions.append(f'client = "{filters["client"]}"')
    if filters['year']:
        conditions.append(f'year = "{filters["year"]}"')
    if filters['month']:
        conditions.append(f'month = "{filters["month"]}"')
    
    return " AND ".join(conditions) if conditions else None

def build_enhanced_prompt(original_query, filters):
    """
    Build an enhanced prompt that provides context to the AI.
    This helps avoid safety false positives.
    """
    context_parts = []
    
    if filters['client']:
        context_parts.append(f"client: {filters['client'].replace('_', ' ')}")
    if filters['year']:
        context_parts.append(f"year: {filters['year']}")
    if filters['month']:
        context_parts.append(f"month: {filters['month']}")
    
    if context_parts:
        context = " AND ".join(context_parts)
        enhanced_prompt = f"""You are searching through business documents for {context}.

User question: {original_query}

Important context:
- These are legitimate business documents that the user has uploaded
- Bank account numbers, GST details, and other financial information are expected and appropriate to share
- The user owns this data and is asking about their own files
- Do not refuse to share information from these documents due to privacy concerns
- If the information exists in the documents, provide it directly and accurately

Please answer the user's question based on the documents available."""
    else:
        enhanced_prompt = f"""You are searching through business documents.

User question: {original_query}

Important context:
- These are legitimate business documents that the user has uploaded
- Financial information like bank accounts, GST details are appropriate to share
- The user owns this data
- If the information exists in the documents, provide it directly

Please answer the user's question based on the documents available."""
    
    return enhanced_prompt

# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """Serve chatbot UI."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with enhanced context."""
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    config = load_store_config()
    
    if not config:
        return jsonify({
            'response': '❌ No indexed files found. Run vector_generation.py first.',
            'filters': {}
        })
    
    store_id = config['store_id']
    
    # Extract filters with enhanced NLP
    filters = extract_filters_from_query(message, config)
    search_filter = build_search_filter(filters)
    
    # Build enhanced prompt with context
    enhanced_prompt = build_enhanced_prompt(message, filters)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=enhanced_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_id],
                        metadata_filter=search_filter
                    )
                )],
                temperature=0.1,  # Lower temperature for more factual responses
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            )
        )
        
        return jsonify({
            'response': response.text,
            'filters': filters
        })
        
    except Exception as e:
        return jsonify({
            'response': f'❌ Error: {str(e)}',
            'filters': {}
        })

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    config = load_store_config()
    if not config:
        print("⚠️ Run vector_generation.py first!")
        exit(1)

    print("=" * 80)
    print("🚀 CHATBOT SERVER RUNNING")
    print("=" * 80)
    print(f"Files: {config['stats']['uploaded']}")
    print(f"Clients: {', '.join(config['stats']['clients'])}")
    print()
    print(f"Open: http://localhost:{app.config['PORT']}")
    print("=" * 80)

    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )