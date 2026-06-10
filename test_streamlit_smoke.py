import os
import sys
import py_compile

def test_streamlit_pages_compile():
    """Verify that all Streamlit app files are syntactically valid and compile successfully."""
    app_files = [
        "app/main.py",
        "app/pages/1_Playground.py",
        "app/pages/2_Benchmark.py"
    ]
    
    for file_path in app_files:
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        # Compile python file to bytecode to check for syntax/import issues
        compiled_path = py_compile.compile(file_path)
        assert compiled_path is not None, f"Failed to compile {file_path}"

if __name__ == "__main__":
    print("Running Streamlit compilation smoke tests...")
    test_streamlit_pages_compile()
    print("All Streamlit files compiled successfully!")
