import os
import sys
import tempfile
import pathlib
import importlib
from unittest.mock import MagicMock, patch
import pytest

def test_config_loads():
    """Verify Config can initialize."""
    from harness.config import AppConfig
    config = AppConfig(GEMINI_API_KEY="dummy_key", DEFAULT_GEMINI_MODEL="gemini-2.5-flash")
    config.validate()
    assert config.GEMINI_API_KEY == "dummy_key"
    assert config.DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"

def test_database_initialization():
    """Verify SQLite schema is created successfully."""
    from harness.database import DatabaseManager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = pathlib.Path(tmpdir) / "test_deploy.db"
        db = DatabaseManager(db_path=str(db_file))
        db.initialize()
        
        assert db_file.exists()
        
        # Verify schema is created
        conn = db._conn
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "benchmark_runs" in tables
        assert "run_logs" in tables
        assert "evaluation_traces" in tables
        db.close()

def test_semantic_model_loads():
    """Verify Semantic Model Warmup executes without crashing."""
    from harness.evaluators.semantic import warmup, SemanticEvaluator
    warmup()
    assert SemanticEvaluator._model is not None

def test_streamlit_pages_import():
    """Verify that all Streamlit pages import successfully using mocked Streamlit."""
    mock_st = MagicMock()
    # Configure selectbox/columns to prevent KeyErrors/unpacking errors
    mock_st.selectbox.side_effect = lambda label, options, *args, **kwargs: options[0] if options else ''
    mock_st.sidebar.selectbox.side_effect = lambda label, options, *args, **kwargs: options[0] if options else ''
    mock_st.columns.side_effect = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    
    # Store originals
    orig_modules = sys.modules.copy()
    
    sys.modules['streamlit'] = mock_st
    sys.modules['plotly'] = MagicMock()
    sys.modules['plotly.graph_objects'] = MagicMock()
    sys.modules['plotly.express'] = MagicMock()
    
    try:
        # Reload/import modules to verify they import successfully
        if 'app.main' in sys.modules:
            importlib.reload(sys.modules['app.main'])
        else:
            importlib.import_module('app.main')
            
        if 'app.pages.1_Playground' in sys.modules:
            importlib.reload(sys.modules['app.pages.1_Playground'])
        else:
            importlib.import_module('app.pages.1_Playground')
            
        if 'app.pages.2_Benchmark' in sys.modules:
            importlib.reload(sys.modules['app.pages.2_Benchmark'])
        else:
            importlib.import_module('app.pages.2_Benchmark')
    except Exception as e:
        pytest.fail(f"Streamlit page import failed: {e}")
    finally:
        # Restore original sys.modules
        for k in list(sys.modules.keys()):
            if k not in orig_modules:
                del sys.modules[k]
        sys.modules.update(orig_modules)

def test_gemini_agent_construction():
    """Verify GeminiAgent can initialize with a mocked API key and no real API calls are made."""
    from harness.agent.gemini_agent import GeminiAgent
    with patch('google.genai.Client') as mock_client:
        agent = GeminiAgent(api_key="mock_key", model_name="gemini-2.5-flash")
        assert agent.api_key == "mock_key"
        assert agent.model_name == "gemini-2.5-flash"
        mock_client.assert_called_once_with(api_key="mock_key")
