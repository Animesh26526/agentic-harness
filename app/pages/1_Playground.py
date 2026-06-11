import streamlit as st
import os
import sys
import uuid
import json
import random
import plotly.graph_objects as go
from pathlib import Path

# Add workspace directory to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from harness.config import Config
from harness.orchestrator import Orchestrator
from harness.database import DatabaseManager
from harness.agent.gemini_agent import GeminiAgent
from harness.scoring import generate_reliability_explanation_markdown

st.set_page_config(
    page_title="Agentic Harness - Playground",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
    /* Styling overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    /* Hide Streamlit right header toolbar components (rerun, settings, screencast, hamburger menu) */
    #MainMenu {
        display: none !important;
    }
    [data-testid="stHeaderDropdownButton"] {
        display: none !important;
    }
    [data-testid="stHeaderRerunButton"] {
        display: none !important;
    }
    button[title="Rerun"] {
        display: none !important;
    }
    button[title="Settings"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Reduce metric font size to prevent clipping */
    [data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium card container */
    .kpi-card {
        background: rgba(30, 34, 53, 0.6);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.6);
    }
    .kpi-val {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 8px 0;
        background: linear-gradient(135deg, #00F2FE 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #A5AEC0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Premium primary buttons styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #000000 !important;
        font-weight: 900 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6) !important;
    }

    .badge {
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .badge-pass {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10B981;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #EF4444;
    }
    .step-header {
        font-weight: bold;
        font-size: 1.1rem;
        color: #E6E8F4;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Interactive Playground")
st.write(
    "Test prompts, configure evaluation criteria, and observe how the self-correcting retry loop "
    "identifies and repairs model failures in real time. Use the quick demo presets below to test instantly."
)

# Sidebar Credentials, Cache, & Glossary
st.sidebar.header("Agent Settings")
api_key_override = st.sidebar.text_input(
    "Gemini API Key override",
    type="password",
    value="",
    placeholder="Configured in .env" if Config.GEMINI_API_KEY else "Enter your API Key...",
    help="Provide Gemini API Key to run real evaluations. If using Llama, please set GROQ_API_KEY or OPENROUTER_API_KEY in .env."
)

st.sidebar.header("Response Cache")
if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
    import importlib
    import harness.cache
    importlib.reload(harness.cache)
    import harness.database
    importlib.reload(harness.database)
    
    cache_manager = harness.cache.ResponseCacheManager()
    cache_manager.clear()
    
    db_manager = harness.database.DatabaseManager()
    db_manager.clear_all_data()
    db_manager.close()
    st.sidebar.success("Cache and performance data cleared successfully!")
    st.rerun()

st.sidebar.header("Reliability Glossary")
st.sidebar.markdown(
    """
    - **Self-Correction**: Closed-loop framework where validation failures compile into repair prompts, instructing the model to self-correct.
    - **Reliability Score**: Composite weighted index combining rule verification (40%) and critic evaluations (60%).
    - **Critic Evaluator**: LLM agent analyzing response logic, constraints, and factual compliance.
    - **Semantic similarity**: Embedding-based validation verifying response factuality against reference ground truth.
    """
)

# Initialize session state variables for presets and evaluations
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'task_category' not in st.session_state:
    st.session_state.task_category = "structured_json"
if 'reference_text' not in st.session_state:
    st.session_state.reference_text = ""
if 'validate_json' not in st.session_state:
    st.session_state.validate_json = False
if 'required_fields' not in st.session_state:
    st.session_state.required_fields = ""
if 'forbidden_keywords' not in st.session_state:
    st.session_state.forbidden_keywords = ""
if 'expand_json' not in st.session_state:
    st.session_state.expand_json = False
if 'expand_keyword' not in st.session_state:
    st.session_state.expand_keyword = False
if 'max_length' not in st.session_state:
    st.session_state.max_length = 0
if 'min_length' not in st.session_state:
    st.session_state.min_length = 0
if 'min_words' not in st.session_state:
    st.session_state.min_words = 0
if 'max_words' not in st.session_state:
    st.session_state.max_words = 0

# Initialize evaluation result in session state
if 'eval_result' not in st.session_state:
    st.session_state.eval_result = None
if 'eval_traces' not in st.session_state:
    st.session_state.eval_traces = []
if 'eval_is_cache_hit' not in st.session_state:
    st.session_state.eval_is_cache_hit = False
if 'eval_harness_enabled' not in st.session_state:
    st.session_state.eval_harness_enabled = False
if 'eval_selected_model' not in st.session_state:
    st.session_state.eval_selected_model = ""
if 'eval_config' not in st.session_state:
    st.session_state.eval_config = {}

# Presets Quick Buttons (Change 4 & Preset UX Improvements)
st.write("")
st.markdown("⚡ **Quick Demo Presets:** Click one to instantly load a test case:")
col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)

with col_p1:
    if st.button("📝 JSON Generation", use_container_width=True):
        st.session_state.user_query = "Generate a customer profile for a client named Alice who is 30 years old and lives in Boston. Output MUST be valid JSON with keys 'name' (string), 'age' (integer), and 'city' (string)."
        st.session_state.task_category = "structured_json"
        st.session_state.validate_json = True
        st.session_state.required_fields = "name, age, city"
        st.session_state.forbidden_keywords = ""
        st.session_state.reference_text = ""
        st.session_state.expand_json = True
        st.session_state.expand_keyword = False
        st.session_state.max_length = 0
        st.session_state.min_length = 0
        st.session_state.min_words = 0
        st.session_state.max_words = 0
        st.session_state.eval_result = None
        st.rerun()
        
with col_p2:
    if st.button("🚫 Constraint Following", use_container_width=True):
        st.session_state.user_query = "Write a short product description of a smartphone. Do NOT use the word 'expensive', 'phone', 'smart', 'device', 'mobile', or 'handset'. Keep the response under 100 characters."
        st.session_state.task_category = "constraint_following"
        st.session_state.validate_json = False
        st.session_state.required_fields = ""
        st.session_state.forbidden_keywords = "expensive, phone, smart, device, mobile, handset"
        st.session_state.reference_text = ""
        st.session_state.expand_json = False
        st.session_state.expand_keyword = True
        st.session_state.max_length = 100
        st.session_state.min_length = 0
        st.session_state.min_words = 0
        st.session_state.max_words = 0
        st.session_state.eval_result = None
        st.rerun()

with col_p3:
    if st.button("📖 Grounded QA", use_container_width=True):
        st.session_state.user_query = "Who developed the Python programming language?"
        st.session_state.task_category = "factual_qa"
        st.session_state.validate_json = False
        st.session_state.required_fields = ""
        st.session_state.forbidden_keywords = ""
        st.session_state.reference_text = "Guido van Rossum developed Python and released it in 1991."
        st.session_state.expand_json = False
        st.session_state.expand_keyword = False
        st.session_state.max_length = 0
        st.session_state.min_length = 0
        st.session_state.min_words = 0
        st.session_state.max_words = 0
        st.session_state.eval_result = None
        st.rerun()

with col_p4:
    if st.button("🔍 Info Extraction", use_container_width=True):
        st.session_state.user_query = "Extract the birth year of Albert Einstein from: Albert Einstein was born on 14 March 1879."
        st.session_state.task_category = "extraction_math"
        st.session_state.validate_json = False
        st.session_state.required_fields = ""
        st.session_state.forbidden_keywords = ""
        st.session_state.reference_text = "1879"
        st.session_state.expand_json = False
        st.session_state.expand_keyword = False
        st.session_state.max_length = 0
        st.session_state.min_length = 0
        st.session_state.min_words = 0
        st.session_state.max_words = 0
        st.session_state.eval_result = None
        st.rerun()

with col_p5:
    if st.button("🔢 Math Reasoning", use_container_width=True):
        st.session_state.user_query = "What is 15 multiplied by 8?"
        st.session_state.task_category = "extraction_math"
        st.session_state.validate_json = False
        st.session_state.required_fields = ""
        st.session_state.forbidden_keywords = ""
        st.session_state.reference_text = "15 * 8 = 120"
        st.session_state.expand_json = False
        st.session_state.expand_keyword = False
        st.session_state.max_length = 0
        st.session_state.min_length = 0
        st.session_state.min_words = 0
        st.session_state.max_words = 0
        st.session_state.eval_result = None
        st.rerun()

with col_p6:
    if st.button("🎲 Random Case", use_container_width=True):
        try:
            with open(PROJECT_ROOT / "data" / "clean_challenge_dataset.json", "r") as f:
                data = json.load(f)
                case = random.choice(data)
                st.session_state.user_query = case.get("prompt", "")
                st.session_state.task_category = case.get("category", "structured_json")
                cfg = case.get("evaluation_config", {})
                st.session_state.validate_json = cfg.get("validate_json", False)
                
                req_fields = cfg.get("required_fields", [])
                st.session_state.required_fields = ", ".join(req_fields) if isinstance(req_fields, list) else str(req_fields)
                
                forb_kw = cfg.get("forbidden_keywords", [])
                st.session_state.forbidden_keywords = ", ".join(forb_kw) if isinstance(forb_kw, list) else str(forb_kw)
                
                st.session_state.reference_text = cfg.get("reference_text", "")
                st.session_state.expand_json = st.session_state.validate_json
                
                st.session_state.max_length = cfg.get("max_length", 0)
                st.session_state.min_length = cfg.get("min_length", 0)
                st.session_state.min_words = cfg.get("min_words", 0)
                st.session_state.max_words = cfg.get("max_words", 0)
                
                st.session_state.expand_keyword = (
                    bool(st.session_state.forbidden_keywords) or 
                    st.session_state.max_length > 0 or 
                    st.session_state.min_length > 0 or 
                    st.session_state.min_words > 0 or 
                    st.session_state.max_words > 0
                )
                
                st.session_state.eval_result = None
                st.rerun()
        except Exception as e:
            st.error(f"Could not load random case: {e}")

st.write("---")

# 🎯 Hardened Challenge Dataset Loading
challenge_dataset_path = PROJECT_ROOT / "data" / "challenge_dataset.json"
if challenge_dataset_path.exists():
    st.markdown("### 🎯 Hardened Challenge Dataset")
    ch_col_sel, ch_col_load = st.columns([4, 1])

    with open(challenge_dataset_path, "r") as f:
        challenge_data = json.load(f)
        
    with ch_col_sel:
        selected_challenge = st.selectbox(
            "Select Challenge Scenario",
            options=challenge_data,
            format_func=lambda x: f"{x['query_id']}: {x['description']}"
        )
    with ch_col_load:
        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_challenge_button = st.button("🎯 Load Challenge", use_container_width=True)
        
    if load_challenge_button:
        ch_config = selected_challenge["evaluation_config"]
        st.session_state.user_query = selected_challenge["input"]
        st.session_state.task_category = selected_challenge["category"]
        st.session_state.validate_json = ch_config.get("validate_json", False)
        req_f = ch_config.get("required_fields", [])
        st.session_state.required_fields = ", ".join(req_f) if isinstance(req_f, list) else str(req_f)
        forb_k = ch_config.get("forbidden_keywords", [])
        st.session_state.forbidden_keywords = ", ".join(forb_k) if isinstance(forb_k, list) else str(forb_k)
        st.session_state.reference_text = ch_config.get("reference_text", selected_challenge.get("expected_output", ""))
        st.session_state.max_length = ch_config.get("max_length", 0)
        st.session_state.min_length = ch_config.get("min_length", 0)
        st.session_state.min_words = ch_config.get("min_words", 0)
        st.session_state.max_words = ch_config.get("max_words", 0)
        
        # Expand sliders/expanders accordingly in session state
        st.session_state.expand_json = bool(ch_config.get("validate_json", False))
        st.session_state.expand_keyword = (
            len(ch_config.get("forbidden_keywords", [])) > 0 
            or ch_config.get("max_length", 0) > 0 
            or ch_config.get("min_length", 0) > 0 
            or ch_config.get("min_words", 0) > 0 
            or ch_config.get("max_words", 0) > 0
        )
        st.rerun()

st.write("---")

# Setup guided columns (Change 2)
g_col_left, g_col_right = st.columns([2, 1])

with g_col_left:
    st.markdown("### 🛠️ Guided Workflow")
    
    # Step 1: Enter Prompt
    st.markdown("<div class='step-header'>Step 1: Enter Prompt</div>", unsafe_allow_html=True)
    user_query = st.text_area(
        "Prompt Query Text",
        value=st.session_state.user_query,
        height=120,
        placeholder="Enter your system prompt here or load a preset..."
    )
    
    # Step 2: Choose Model
    st.markdown("<div class='step-header'>Step 2: Choose Model</div>", unsafe_allow_html=True)
    model_options = [
        "Gemma 4 26B",
        "Gemma 4 31B",
        "Llama 3.1 8B Instant",
        "Gemini 2.5 Flash"
    ]
    selected_model = st.selectbox(
        "Target LLM Backend",
        options=model_options,
        index=0,
        help="Select the LLM model to query. Caching and provider abstraction apply dynamically."
    )
    
    # Step 3: Choose Harness Mode
    st.markdown("<div class='step-header'>Step 3: Choose Harness Mode</div>", unsafe_allow_html=True)
    harness_mode = st.radio(
        "Self-Correction Verification Loop",
        options=["OFF (Direct Generation)", "ON (Verify & Self-Correct)"],
        index=1,
        horizontal=True,
        help="ON compiles evaluations and triggers self-correcting repair loops. OFF runs raw generation."
    )
    harness_enabled = ("ON" in harness_mode)

with g_col_right:
    st.markdown("### 📋 Verification & Evaluator Rules")
    st.write("Configure targets to check against during the evaluation pipeline.")
    
    task_category = st.selectbox(
        "Task Category Type",
        ["structured_json", "constraint_following", "factual_qa", "extraction_math"],
        index=["structured_json", "constraint_following", "factual_qa", "extraction_math"].index(st.session_state.task_category)
    )

    reference_text_input = st.text_input(
        "Reference Text / Ground Truth",
        value=st.session_state.reference_text,
        placeholder="Required for factual_qa/extraction_math"
    )

    # Preset UX Improvements: expand relevant expander automatically
    with st.expander("Structured JSON Validation Settings", expanded=bool(st.session_state.expand_json)):
        is_json_active = bool(st.session_state.validate_json)
        validate_json_checked = st.checkbox("Enforce JSON Schema Check", value=is_json_active)
        if not validate_json_checked:
            validate_json = False
        else:
            if st.session_state.validate_json == "invalid":
                validate_json = "invalid"
            else:
                validate_json = True
        required_fields_input = st.text_input(
            "Required JSON Keys (comma-separated)",
            value=st.session_state.required_fields,
            placeholder="e.g. name, age, city"
        )

    with st.expander("Keyword & Length Constraint Settings", expanded=st.session_state.expand_keyword):
        forbidden_keywords_input = st.text_input(
            "Forbidden Words (comma-separated)",
            value=st.session_state.forbidden_keywords,
            placeholder="e.g. expensive, invalid"
        )
        max_length_input = st.number_input(
            "Maximum Character Length (0 for no limit)",
            value=st.session_state.max_length,
            min_value=0,
            help="Strict limit checked by the rule-based validator."
        )
        min_length_input = st.number_input(
            "Minimum Character Length (0 for no limit)",
            value=st.session_state.min_length,
            min_value=0,
            help="Strict minimum character limit checked by the rule-based validator."
        )
        min_words_input = st.number_input(
            "Minimum Word Count (0 for no limit)",
            value=st.session_state.min_words,
            min_value=0,
            help="Strict minimum word count limit checked by the rule-based validator."
        )
        max_words_input = st.number_input(
            "Maximum Word Count (0 for no limit)",
            value=st.session_state.max_words,
            min_value=0,
            help="Strict maximum word count limit checked by the rule-based validator."
        )

# Format lists
def parse_comma_separated_input(val):
    if not val:
        return []
    cleaned = str(val).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    items = []
    for item in cleaned.split(","):
        item = item.strip().strip("'").strip('"')
        if item:
            items.append(item)
    return items

required_fields = parse_comma_separated_input(required_fields_input)
forbidden_keywords = parse_comma_separated_input(forbidden_keywords_input)

try:
    max_length_val = int(max_length_input)
except (ValueError, TypeError):
    max_length_val = 0

try:
    min_length_val = int(min_length_input)
except (ValueError, TypeError):
    min_length_val = 0

try:
    min_words_val = int(min_words_input)
except (ValueError, TypeError):
    min_words_val = 0

try:
    max_words_val = int(max_words_input)
except (ValueError, TypeError):
    max_words_val = 0

evaluation_config = {
    "validate_json": validate_json,
    "required_fields": required_fields,
    "forbidden_keywords": forbidden_keywords,
    "reference_text": reference_text_input,
    "max_length": max_length_val if max_length_val > 0 else None,
    "min_length": min_length_val if min_length_val > 0 else None,
    "min_words": min_words_val if min_words_val > 0 else None,
    "max_words": max_words_val if max_words_val > 0 else None
}

# Auto-inject type mappings for JSON profile presets
if task_category == "structured_json" and "name" in required_fields and "age" in required_fields:
    evaluation_config["field_types"] = {
        "name": "string",
        "age": "integer",
        "city": "string"
    }

# Step 4: Run Evaluation
with g_col_left:
    st.write("---")
    st.markdown("### 🔍 Active Constraints Preview")
    st.markdown("These are the strict constraints the Evaluator and Critic will use to judge the response:")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if evaluation_config.get("max_length"):
            st.markdown(f"- **Max Character Limit:** {evaluation_config['max_length']}")
        if evaluation_config.get("min_length"):
            st.markdown(f"- **Min Character Limit:** {evaluation_config['min_length']}")
        if evaluation_config.get("max_words"):
            st.markdown(f"- **Max Words:** {evaluation_config['max_words']}")
        if evaluation_config.get("min_words"):
            st.markdown(f"- **Min Words:** {evaluation_config['min_words']}")
        if evaluation_config.get("validate_json"):
            st.markdown(f"- **JSON Validation:** ON")
    with col_c2:
        if evaluation_config.get("forbidden_keywords"):
            st.markdown(f"- **Forbidden Keywords:** {', '.join(evaluation_config['forbidden_keywords'])}")
        if evaluation_config.get("required_fields"):
            st.markdown(f"- **Required Fields:** {', '.join(evaluation_config['required_fields'])}")
            
    if not any([evaluation_config.get("max_length"), evaluation_config.get("min_length"), evaluation_config.get("max_words"), evaluation_config.get("min_words"), evaluation_config.get("validate_json"), evaluation_config.get("forbidden_keywords"), evaluation_config.get("required_fields")]):
        st.info("No objective constraints configured. Only factual/semantic coherence will be evaluated.")
        
    st.write("")
    submit_col1, submit_col2 = st.columns([3, 1])
    with submit_col1:
        submit = st.button("🚀 Step 4: Execute Playground Evaluation", type="primary", use_container_width=True)
    with submit_col2:
        try:
            default_pacing = float(os.getenv("GEMINI_PACING_DELAY", "2.0"))
        except ValueError:
            default_pacing = 2.0
        sleep_delay = st.slider(
            "Pacing Delay (sec)",
            0.0, 10.0, default_pacing, 0.5,
            help="Central pacing between API calls to respect rate limits."
        )

if submit:
    active_key = api_key_override.strip() or Config.GEMINI_API_KEY
    if not user_query.strip():
        st.warning("Please enter a query prompt before submitting.")
    elif not active_key and "llama" not in selected_model.lower():
        st.error("Missing Gemini API Key. Please provide your Gemini API key in the sidebar or configure it in your .env file.")
    else:
        run_id = f"play_run_{uuid.uuid4().hex[:6]}"
        query_id = f"play_q_{uuid.uuid4().hex[:6]}"

        db_path = str(PROJECT_ROOT / "harness_metrics.db")
        db_manager = DatabaseManager(db_path=db_path)
        db_manager.initialize()

        with st.spinner("Executing evaluation pipeline..."):
            try:
                # 1. Check SQLite Response Cache (Change 6)
                from harness.cache import ResponseCacheManager
                cache_manager = ResponseCacheManager()
                
                cached_data = cache_manager.get(user_query, selected_model, harness_enabled)
                
                result = None
                traces = []
                
                if cached_data:
                    from harness.orchestrator import ExecutionResult
                    result = ExecutionResult(
                        query_id=cached_data["query_id"],
                        category=cached_data["category"],
                        raw_response=cached_data["raw_response"],
                        final_response=cached_data["final_response"],
                        semantic_score=cached_data.get("semantic_score"),
                        rule_score=cached_data.get("rule_score"),
                        critic_score=cached_data.get("critic_score"),
                        overall_score=cached_data["overall_score"],
                        retry_count=cached_data["retry_count"],
                        passed=cached_data["passed"],
                        issues=cached_data["issues"]
                    )
                    traces = cached_data.get("traces", [])
                else:
                    # Configure dynamic delay/model in env
                    os.environ["GEMINI_PACING_DELAY"] = str(sleep_delay)
                    os.environ["DEFAULT_GEMINI_MODEL"] = selected_model
                    
                    # Run live
                    agent = GeminiAgent(api_key=active_key, model_name=selected_model)
                    orchestrator = Orchestrator(agent=agent, db_manager=db_manager)
                
                    result = orchestrator.execute(
                        query=user_query,
                        category=task_category,
                        evaluation_config=evaluation_config,
                        harness_enabled=harness_enabled,
                        run_id=run_id,
                        query_id=query_id
                    )
                    
                    # Fetch traces
                    raw_traces = db_manager.get_traces(run_id, query_id)
                    traces = [dict(t) for t in raw_traces]
                    
                    # Save Cache (Change 6)
                    cache_data = {
                        "query_id": result.query_id,
                        "category": result.category,
                        "raw_response": result.raw_response,
                        "final_response": result.final_response,
                        "semantic_score": result.semantic_score,
                        "rule_score": result.rule_score,
                        "critic_score": result.critic_score,
                        "overall_score": result.overall_score,
                        "retry_count": result.retry_count,
                        "passed": result.passed,
                        "issues": result.issues,
                        "traces": traces
                    }
                    cache_manager.set(user_query, selected_model, harness_enabled, cache_data)
                
                # Render UI Outcomes (Change 2)
                st.write("---")
                
                # Caching Effectiveness Metrics (Cache UI Validation)
                st.markdown("### ⚡ Performance & Caching metrics")
                
                # Render Cache Criteria & Execution Source
                st.markdown(f"**Execution Source**: `{'Cache' if cached_data else 'Live API'}`")
                st.markdown(f"**Cache Criteria**: Query + Model (`{selected_model}`) + Harness (`{'ON' if harness_enabled else 'OFF'}`)")
                
                c_col1, c_col2, c_col3 = st.columns(3)
                
                if cached_data:
                    # Cache Hit display
                    saved_calls = 1 + result.retry_count
                    saved_latency = saved_calls * 2.5  # Estimate 2.5s saved per LLM call
                    with c_col1:
                        st.markdown("<span class='badge badge-pass' style='font-size:1.1rem; padding:8px 16px;'>🟢 CACHE HIT</span>", unsafe_allow_html=True)
                    with c_col2:
                        st.metric("Saved API Calls", f"{saved_calls}", help="Number of model generation or critic evaluator queries bypassed.")
                    with c_col3:
                        st.metric("Saved Latency Estimate", f"{saved_latency:.1f}s", help="Estimated network and inference duration avoided.")
                else:
                    # Cache Miss display
                    calls_made = 1 + result.retry_count
                    with c_col1:
                        st.markdown("<span class='badge badge-fail' style='background:rgba(59,130,246,0.2); border:1px solid rgba(59,130,246,0.4); color:#3B82F6; font-size:1.1rem; padding:8px 16px;'>🔵 CACHE MISS</span>", unsafe_allow_html=True)
                    with c_col2:
                        st.metric("API Calls Executed", f"{calls_made}", help="Total model generations and critic evaluations executed.")
                    with c_col3:
                        st.metric("Response Indexed", "Success", help="Response and full validation traces written to SQLite cache database.")

                st.write("")

                # LEFT vs RIGHT Outputs
                out_col_left, out_col_right = st.columns(2)
                with out_col_left:
                    st.markdown("### 📥 Raw Model Output")
                    st.info(result.raw_response if result.raw_response else "(Empty response)")
                    
                with out_col_right:
                    st.markdown("### 🛡️ Harness Output")
                    if harness_enabled:
                        st.success(result.final_response if result.final_response else "(Empty response)")
                    else:
                        st.warning(result.final_response if result.final_response else "(Empty response)")
                
                # Below: Evaluation Breakdown (Change 2 & Change 3 tooltips)
                st.write("")
                st.markdown("### 📊 Evaluation Breakdown")
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                
                with b_col1:
                    score_str = f"{result.semantic_score:.3f}" if result.semantic_score is not None else "N/A"
                    st.markdown(f"""
                    <div class="kpi-card" title="Embedding similarity measuring how well the meaning matches the ground truth.">
                        <div class="kpi-label">Semantic Score</div>
                        <div class="kpi-val">{score_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with b_col2:
                    score_str = f"{result.rule_score:.3f}" if result.rule_score is not None else "N/A"
                    st.markdown(f"""
                    <div class="kpi-card" title="Verification of rigid keywords, length limits, or JSON syntax.">
                        <div class="kpi-label">Rule Score</div>
                        <div class="kpi-val">{score_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with b_col3:
                    score_str = f"{result.critic_score:.3f}" if result.critic_score is not None else "N/A"
                    st.markdown(f"""
                    <div class="kpi-card" title="Model-graded evaluator checking instruction following details.">
                        <div class="kpi-label">Critic Score</div>
                        <div class="kpi-val">{score_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with b_col4:
                    if not harness_enabled:
                        rel_str = "N/A"
                    else:
                        rel_str = f"{int(result.overall_score * 100)}/100"
                        
                    st.markdown(f"""
                    <div class="kpi-card" title="Composite score representing final overall compliance rate.">
                        <div class="kpi-label">Reliability Score</div>
                        <div class="kpi-val">{rel_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if harness_enabled:
                    issues_list = result.issues if isinstance(result.issues, list) else []
                    if isinstance(result.issues, str):
                        try:
                            issues_list = json.loads(result.issues)
                        except:
                            try:
                                issues_list = eval(result.issues)
                            except:
                                issues_list = []
                                
                    explanation_html = generate_reliability_explanation_markdown(
                        overall_score=result.overall_score,
                        passed=result.passed,
                        category=result.category,
                        rule_score=result.rule_score,
                        semantic_score=result.semantic_score,
                        critic_score=result.critic_score,
                        issues=issues_list,
                        retry_count=result.retry_count,
                        evaluation_config=evaluation_config
                    )
                    st.markdown(explanation_html, unsafe_allow_html=True)
                
                # Below: Retry Trace Timeline (Change 2)
                if harness_enabled and len(traces) > 0:
                    st.write("")
                    st.markdown("### 🔄 Retry Trace Timeline")
                    
                    for idx, trace in enumerate(traces):
                        t_verdict = "PASSED" if trace["overall_reliability"] >= Config.RELIABILITY_THRESHOLD else "FAILED"
                        expander_title = f"Attempt {trace['attempt']} — Score: {trace['overall_reliability']:.3f} [{t_verdict}]"
                        
                        with st.expander(expander_title, expanded=(idx == len(traces)-1)):
                            st.markdown("**Output Generated:**")
                            st.code(trace["raw_response"])
                            
                            st.markdown("**Scores Computed:**")
                            sc_col1, sc_col2, sc_col3 = st.columns(3)
                            with sc_col1:
                                st.write(f"- Semantic Score: `{trace['semantic_score']}`")
                            with sc_col2:
                                st.write(f"- Rule Score: `{trace['rule_score']}`")
                            with sc_col3:
                                st.write(f"- Critic Score: `{trace['critic_score']}`")
                            
                            # Parse issues
                            issues_list = []
                            if trace.get("issues"):
                                try:
                                    issues_list = json.loads(trace["issues"]) if isinstance(trace["issues"], str) else trace["issues"]
                                except Exception:
                                    try:
                                        issues_list = eval(trace["issues"]) if isinstance(trace["issues"], str) else trace["issues"]
                                    except Exception:
                                        issues_list = []
                            
                            if issues_list:
                                st.markdown("<span style='color:#EF4444; font-weight:bold;'>Violations:</span>", unsafe_allow_html=True)
                                for issue in issues_list:
                                    st.checkbox(f"❌ {issue}", value=False, disabled=True, key=f"issue_{idx}_{uuid.uuid4().hex[:8]}")
                            else:
                                st.checkbox("✅ All verification rules passed successfully.", value=True, disabled=True, key=f"pass_{idx}")
                            
                            # Critic Transparency (Expose Critic Rationale/Reasoning)
                            if trace.get("critic_feedback"):
                                st.write("")
                                with st.expander("🔍 Show Critic Evaluator Rationale (Raw LLM Grade)", expanded=False):
                                    st.text_area("Raw Critic Critique JSON", value=trace["critic_feedback"], height=120, disabled=True, key=f"critic_feedback_{idx}")

                            if trace.get("retry_triggered") == 1:
                                st.write("🔄 Retry loop triggered with compiler feedback.")

                # Below: Agentic Evaluation Narrative
                st.write("")
                st.markdown("### 💡 Agentic Evaluation Narrative")
                if harness_enabled:
                    if result.retry_count > 0:
                        initial_issues = []
                        if len(traces) > 0 and traces[0].get("issues"):
                            try:
                                initial_issues = json.loads(traces[0]["issues"]) if isinstance(traces[0]["issues"], str) else traces[0]["issues"]
                            except:
                                try:
                                    initial_issues = eval(traces[0]["issues"]) if isinstance(traces[0]["issues"], str) else traces[0]["issues"]
                                except:
                                    initial_issues = []
                        
                        outstanding_issues = []
                        if len(traces) > 0 and traces[-1].get("issues"):
                            try:
                                outstanding_issues = json.loads(traces[-1]["issues"]) if isinstance(traces[-1]["issues"], str) else traces[-1]["issues"]
                            except:
                                try:
                                    outstanding_issues = eval(traces[-1]["issues"]) if isinstance(traces[-1]["issues"], str) else traces[-1]["issues"]
                                except:
                                    outstanding_issues = []
                        resolved_issues = [iss for iss in initial_issues if iss not in outstanding_issues]
                        
                        outcome_str = "Passed" if result.passed else "Failed"
                        rel_str = f"{int(result.overall_score * 100)} / 100"
                        
                        explanation = f"**Outcome:** {outcome_str}  \n"
                        explanation += f"**Reliability:** {rel_str}  \n"
                        explanation += f"**Retry Count:** {result.retry_count}  \n\n"
                        
                        explanation += "**Initial Violations Detected:**  \n"
                        if initial_issues:
                            for issue in initial_issues:
                                explanation += f"- {issue}  \n"
                        else:
                            explanation += "- None  \n"
                            
                        explanation += "\n**Resolved:**  \n"
                        if resolved_issues:
                            for issue in resolved_issues:
                                explanation += f"- ✅ {issue}  \n"
                        else:
                            explanation += "- None  \n"
                            
                        explanation += "\n**Remaining Issues:**  \n"
                        if outstanding_issues:
                            for issue in outstanding_issues:
                                explanation += f"- ❌ {issue}  \n"
                        else:
                            explanation += "- None  \n"

                        if result.passed:
                            st.success(explanation)
                        else:
                            st.error(explanation)
                    else:
                        outcome_str = "Passed" if result.passed else "Failed"
                        rel_str = f"{int(result.overall_score * 100)} / 100"
                        
                        explanation = f"**Outcome:** {outcome_str}  \n"
                        explanation += f"**Reliability:** {rel_str}  \n"
                        explanation += f"**Retry Count:** 0  \n\n"
                        explanation += "No correction was needed! The raw model output passed all safety, semantic, and rule-based checks on the first attempt."
                        
                        if result.passed:
                            st.success(explanation)
                        else:
                            st.error(explanation)
                else:
                    st.warning("Self-correction was disabled (Harness OFF). The response was returned directly from the model without verification checks.")

            except Exception as e:
                st.error(f"Playground execution failed: {str(e)}")
            finally:
                db_manager.close()

# Glossary at bottom of page
st.write("---")
with st.expander("📖 Reliability Glossary Definitions", expanded=False):
    st.markdown("""
    - **Reliability Score**: A composite weighted index combining 40% Objective Rules and 60% Subjective Critic LLM Grade.
    - **Objective Score**: Pass/Fail metrics from deterministic parsing checks (regex, length, JSON syntax).
    - **Subjective Score**: Fine-grained semantic grading and instruction-following checks by a Critic LLM.
    """)
