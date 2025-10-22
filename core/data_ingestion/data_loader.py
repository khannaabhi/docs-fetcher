import os
import glob # For finding files in directories
import shutil # For removing directories

import os
import shutil

def populate_dummy_data(data_dir="docs", lancedb_path="lancedb_rag_data"):
    """
    Creates a dummy nested directory structure with sample Markdown files.
    Also cleans up existing LanceDB data for a fresh start.
    """
    # Clean up existing LanceDB data
    if os.path.exists(lancedb_path):
        print(f"Removing existing LanceDB directory: {lancedb_path}")
        shutil.rmtree(lancedb_path)
        print("Existing LanceDB directory removed.")

#     # Create dummy data directory and files
#     print(f"\nCreating dummy data in: {data_dir}")
#     os.makedirs(os.path.join(data_dir, "project_a"), exist_ok=True)
#     os.makedirs(os.path.join(data_dir, "project_b", "sub_project"), exist_ok=True)

#     with open(os.path.join(data_dir, "project_a", "README.md"), "w") as f:
#         f.write("""# Project A Overview
# This is the main README for Project A. It focuses on developing a new machine learning algorithm for anomaly detection in time series data.
# The algorithm uses a combination of recurrent neural networks and statistical methods.
# Key features include real-time processing and high accuracy.
# """)

#     with open(os.path.join(data_dir, "project_a", "INSTALL.md"), "w") as f:
#         f.write("""# Installation Guide for Project A
# To install Project A, ensure you have Python 3.9+ and pip.
# 1. Clone the repository: `git clone https://github.com/yourorg/project_a.git`
# 2. Navigate to the directory: `cd project_a`
# 3. Install dependencies: `pip install -r requirements.txt`
# 4. Run tests: `pytest`
# """)

#     with open(os.path.join(data_dir, "project_b", "README.md"), "w") as f:
#         f.write("""# Project B: Data Visualization Tool
# Project B is a web-based tool for visualizing large datasets. It supports various chart types including bar charts, line graphs, and scatter plots.
# It's built using React for the frontend and FastAPI for the backend.
# Users can upload CSV files and interactively explore their data.
# """)

#     with open(os.path.join(data_dir, "project_b", "sub_project", "README.md"), "w") as f:
#         f.write("""# Sub-Project within Project B: Real-time Dashboard
# This sub-project focuses on building a real-time dashboard component for Project B.
# It uses WebSockets to push live data updates to the client.
# Technologies involved are D3.js for interactive elements and Redis for caching.
# """)
#     print("Dummy data populated successfully.")



def read_markdown_files(directory):
    """Reads all .md files from a directory and its subdirectories."""
    file_contents = []
    file_paths = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Store content along with its original file path for better context
                file_contents.append({"text": content, "source_path": file_path})
            print(f"Read: {file_path}")
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
    return file_contents
