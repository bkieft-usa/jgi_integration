import os
import sys
import yaml
import glob
import pandas as pd
import numpy as np
import shutil
from typing import Dict, Any, List, Optional, Tuple
from IPython.display import display, Javascript
from IPython import get_ipython
import tools.helpers as hlp
import logging
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import inspect
import re
import time
import hashlib
import duckdb
import openai
import json
from pathlib import Path

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "\033[47m%(levelname)s - %(message)s\033[0m"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

class ConfigManager:
    """Manages configuration files with automatic hash-based tagging."""
    
    def __init__(self, config_dir: str, data_processing_hash: str = None, analysis_hash: str = None):
        self.config_dir = Path(glob.glob(config_dir)[0])
        self.target_data_hash = data_processing_hash
        self.target_analysis_hash = analysis_hash
        
        # Set up config file paths
        self.project_config_file = self.config_dir / "project.yml"
        self.data_processing_config_file = self.config_dir / "data_processing.yml"
        self.analysis_config_file = self.config_dir / "analysis.yml"
        
        # If hashes are specified, try to load from persistent configs
        if self.target_data_hash and self.target_analysis_hash:
            self._load_from_persistent_configs()
        elif not self._standard_configs_exist():
            log.warning("Standard config files not found and no hashes specified.")
    
    def _standard_configs_exist(self) -> bool:
        """Check if standard config files exist."""
        return all([
            self.project_config_file.exists(),
            self.data_processing_config_file.exists(),
            self.analysis_config_file.exists()
        ])
    
    def _load_from_persistent_configs(self):
        """Load from persistent config files with specified hashes."""
        log.info(f"Loading persistent configs for hashes: {self.target_data_hash}, {self.target_analysis_hash}")
        
        # Look for persistent config files with the exact hash pattern
        pattern = f"Dataset_Processing--{self.target_data_hash}_Analysis--{self.target_analysis_hash}_*_config.yml"
        config_files = list(self.config_dir.glob(pattern))
        
        if not config_files:
            log.error(f"No persistent config files found for pattern: {pattern}")
            raise FileNotFoundError(f"Cannot find configs for the specified hashes in {self.config_dir}")
        
        # Group by config type
        config_mapping = {
            'project': self.project_config_file,
            'data': self.data_processing_config_file,
            'analysis': self.analysis_config_file
        }
        
        found_configs = {}
        for config_file in config_files:
            name_parts = config_file.stem.split('_')
            if len(name_parts) >= 3:
                config_type = name_parts[-2]  # e.g., 'project', 'data', 'analysis'
                
                # Handle both 'data' and 'data_processing' naming
                if config_type == 'data':
                    config_type = 'data'
                elif config_type == 'processing':
                    config_type = 'data'
                
                if config_type in config_mapping:
                    found_configs[config_type] = config_file
        
        # Check if we have all required configs
        required_types = ['project', 'data', 'analysis']
        missing_types = [t for t in required_types if t not in found_configs]
        
        if missing_types:
            log.error(f"Missing config types: {missing_types}")
            raise FileNotFoundError(f"Cannot find all required config types for specified hashes")
        
        # Copy persistent configs to standard names
        for config_type, target_file in config_mapping.items():
            if config_type in found_configs:
                source_file = found_configs[config_type]
                shutil.copy2(source_file, target_file)
                log.info(f"Loaded {config_type} config from: {source_file.name}")
    
    def load_configs(self) -> Tuple[Dict[str, Any], str, str]:
        """Load all config files and generate hashes."""
        
        # Load project config
        with open(self.project_config_file, 'r') as f:
            project_config = yaml.safe_load(f)
        
        # Load data processing config and generate hash
        with open(self.data_processing_config_file, 'r') as f:
            data_processing_config = yaml.safe_load(f)
        data_processing_hash = self._generate_config_hash(data_processing_config)
        
        # Load analysis config and generate hash
        with open(self.analysis_config_file, 'r') as f:
            analysis_config = yaml.safe_load(f)
        analysis_hash = self._generate_config_hash(analysis_config)
        
        # Verify hashes match if they were specified
        if self.target_data_hash and data_processing_hash != self.target_data_hash:
            log.warning(f"Data processing hash mismatch! Expected: {self.target_data_hash}, Got: {data_processing_hash}")

        if self.target_analysis_hash and analysis_hash != self.target_analysis_hash:
            log.warning(f"Analysis hash mismatch! Expected: {self.target_analysis_hash}, Got: {analysis_hash}")
        
        # Combine all configs for backward compatibility
        combined_config = {
            **project_config,
            **data_processing_config,
            **analysis_config
        }
        
        return combined_config, data_processing_hash, analysis_hash
    
    def _generate_config_hash(self, config: Dict[str, Any], length: int = 8) -> str:
        """Generate a deterministic hash from config parameters."""
        # Convert to JSON string with sorted keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        full_hash = hash_obj.hexdigest()
        
        # Return truncated hash
        return full_hash[:length]
    
    def check_config_changes(self, previous_data_hash: Optional[str] = None, 
                           previous_analysis_hash: Optional[str] = None) -> Dict[str, bool]:
        """Check if configs have changed compared to previous hashes."""
        _, current_data_hash, current_analysis_hash = self.load_configs()
        
        changes = {
            'data_processing_changed': previous_data_hash != current_data_hash,
            'analysis_changed': previous_analysis_hash != current_analysis_hash,
            'current_data_hash': current_data_hash,
            'current_analysis_hash': current_analysis_hash
        }
        
        return changes
    
    def get_hash_info(self) -> Dict[str, str]:
        """Get current hash information for both configs."""
        _, data_hash, analysis_hash = self.load_configs()
        return {
            'data_processing_hash': data_hash,
            'analysis_hash': analysis_hash
        }

class DataRegistry:
    """Registry that automatically syncs DataFrames to DuckDB for AI querying."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)
        self.table_metadata = {}
        self._setup_schema()
    
    def _setup_schema(self):
        """Create metadata tables for tracking datasets and analyses."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_catalog (
                table_name VARCHAR PRIMARY KEY,
                object_type VARCHAR,  -- 'dataset' or 'analysis'
                object_name VARCHAR,  -- dataset_name or analysis identifier
                attribute_name VARCHAR,
                description VARCHAR,
                shape_rows INTEGER,
                shape_cols INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str, 
                        object_type: str, object_name: str, 
                        attribute_name: str, description: str = ""):
        """Register a DataFrame in DuckDB with metadata."""
        if df.empty:
            return
        
        # Check if the index contains meaningful data (not just default numeric index)
        has_meaningful_index = not df.index.equals(pd.RangeIndex(len(df)))
        
        # Store the DataFrame with index preserved if meaningful
        if has_meaningful_index:
            # Reset index to make it a column, then register
            df_with_index = df.reset_index()
            self.conn.register(table_name, df_with_index)
        else:
            # Store DataFrame as-is
            self.conn.register(table_name, df)
        
        # Update catalog with index information
        self.conn.execute("""
            INSERT OR REPLACE INTO data_catalog 
            (table_name, object_type, object_name, attribute_name, description, shape_rows, shape_cols)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (table_name, object_type, object_name, attribute_name, description, len(df), len(df.columns)))
        
        self.table_metadata[table_name] = {
            'object_type': object_type,
            'object_name': object_name,
            'attribute_name': attribute_name,
            'columns': list(df_with_index.columns if has_meaningful_index else df.columns),
            'dtypes': {col: str(dtype) for col, dtype in (df_with_index.dtypes if has_meaningful_index else df.dtypes).items()},
            'has_meaningful_index': has_meaningful_index,
            'index_name': df.index.name if has_meaningful_index else None
        }
    
    def get_schema_info(self) -> str:
        """Get human-readable schema information for the AI agent."""
        catalog_df = self.conn.execute("SELECT * FROM data_catalog ORDER BY object_type, object_name").df()
        
        schema_info = "Available Tables:\n\n"
        for _, row in catalog_df.iterrows():
            table_name = row['table_name']
            metadata = self.table_metadata.get(table_name, {})
            
            schema_info += f"Table: {table_name}\n"
            schema_info += f"  - Type: {row['object_type']} ({row['object_name']})\n"
            schema_info += f"  - Attribute: {row['attribute_name']}\n"
            schema_info += f"  - Shape: {row['shape_rows']} rows by {row['shape_cols']} columns\n"
            schema_info += f"  - Description: {row['description']}\n"
            
            if 'columns' in metadata:
                schema_info += f"  - Columns: {', '.join(metadata['columns'])}\n"
            schema_info += "\n"
        
        return schema_info

class AIQueryAgent:
    """AI agent that converts natural language to SQL queries."""
    
    def __init__(self, data_registry: DataRegistry, openai_api_key: str = None):
        self.data_registry = data_registry
        self.client = None
        if openai_api_key:
            # Initialize the OpenAI client for CBorg with the new API
            self.client = openai.OpenAI(api_key=openai_api_key, base_url="https://api.cborg.lbl.gov")
    
    def query(self, natural_language_query: str, limit: int = 500) -> pd.DataFrame:
        """Convert natural language to SQL and execute."""
        try:
            sql_query = self._generate_sql(natural_language_query)
            print(f"Generated SQL: {sql_query}")
            
            # Add safety limit
            if "LIMIT" not in sql_query.upper():
                sql_query += f" LIMIT {limit}"
            
            result = self.data_registry.conn.execute(sql_query).df()
            return result
            
        except Exception as e:
            print(f"Query failed: {e}")
            return pd.DataFrame()
    
    def _generate_sql(self, query: str) -> str:
        """Generate SQL from natural language using AI."""
        schema_info = self.data_registry.get_schema_info()
        
        prompt = f"""
You are a SQL expert. Convert the following natural language query to SQL using DuckDB syntax.

Database Schema:
{schema_info}

User Query: "{query}"

Rules:
1. Use exact table and column names from the schema
2. Return only the SQL query, no explanations
3. Use proper DuckDB syntax
4. Handle common variations in natural language input appropriately
5. If filtering by numeric values, use appropriate comparison operators
6. IMPORTANT: Submodule values are stored as strings like 'submodule_1', 'submodule_15', etc. When filtering by submodule number, use the format 'submodule_X' where X is the number
7. Do NOT wrap the response in markdown code blocks or backticks
8. For p-value or statistical significance queries, look for columns containing 'pvalue', 'p_value', 'corrected_pvalue', or similar
9. When filtering by text content "containing" or "with phrase", use LIKE operator with wildcards (e.g., column LIKE '%text%')
10. For "not equal" conditions like "not 'Unassigned'", use != or <> operator
11. When querying correlation data, look for columns like 'correlation', 'corr_score', or similar numerical correlation measures
12. For network/node queries, prioritize tables with 'node' in the name over 'edge' tables
13. When filtering by annotation classes or categories, look for columns ending in '_class', '_category', '_annotation', or '_superclass'
14. Handle case-insensitive text matching when appropriate using UPPER() or LOWER() functions

Common Query Patterns:
- "submodule X" → WHERE submodule = 'submodule_X'
- "correlation > 0.7" → WHERE correlation > 0.7 (or similar correlation column)
- "p value < 0.05" → WHERE pvalue < 0.05 (or corrected_pvalue, etc.)
- "contains 'text'" → WHERE column LIKE '%text%'
- "not 'value'" → WHERE column != 'value'

SQL Query:
"""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="openai/chatgpt:latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                sql_query = response.choices[0].message.content.strip()
                
                # Clean up any markdown formatting that might still appear
                sql_query = self._clean_sql_response(sql_query)
                
                return sql_query
            except Exception as e:
                raise ValueError(f"OpenAI API call failed: {e}")
    
    def _clean_sql_response(self, sql_response: str) -> str:
        """Clean up SQL response from AI to remove markdown formatting."""
        # Remove markdown code blocks
        if sql_response.startswith('```sql'):
            sql_response = sql_response[6:]
        elif sql_response.startswith('```'):
            sql_response = sql_response[3:]
        
        if sql_response.endswith('```'):
            sql_response = sql_response[:-3]
        
        # Remove any remaining backticks
        sql_response = sql_response.replace('`', '')
        
        # Clean up whitespace
        sql_response = sql_response.strip()
        
        # Remove trailing semicolon to prevent syntax errors when adding LIMIT
        if sql_response.endswith(';'):
            sql_response = sql_response[:-1]
        
        return sql_response

class WorkflowProgressTracker:
    """Class to track and visualize workflow progress."""
    
    def __init__(self, project_name="JGI Integration Workflow"):
        self.project_name = project_name
        self.current_step = None
        self.completed_steps = set()
        self.steps = []
        self.step_mapping = {}
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Initialize the workflow steps."""
        # Define steps in linear order
        self.steps = [
            {'id': 'init_project', 'label': 'Initialize Project', 'category': 'setup'},
            {'id': 'create_datasets', 'label': 'Create Dataset Objects', 'category': 'setup'},
            {'id': 'create_analysis', 'label': 'Create Analysis Object', 'category': 'setup'},
            {'id': 'link_metadata', 'label': 'Link Metadata', 'category': 'data_processing'},
            {'id': 'link_data', 'label': 'Link Data', 'category': 'data_processing'},
            {'id': 'filter_dataset_features', 'label': 'Filter Rare Features', 'category': 'data_processing'},
            {'id': 'devariance_dataset_features', 'label': 'Remove Low-Variance Features', 'category': 'data_processing'},
            {'id': 'scale_dataset_features', 'label': 'Scale Features', 'category': 'data_processing'},
            {'id': 'replicability_test_dataset_features', 'label': 'Replicability Filtering', 'category': 'data_processing'},
            {'id': 'integrate_metadata', 'label': 'Integrate Metadata', 'category': 'analysis'},
            {'id': 'integrate_data', 'label': 'Integrate Data', 'category': 'analysis'},
            {'id': 'feature_selection', 'label': 'Feature Selection', 'category': 'analysis'},
            {'id': 'calculate_correlations', 'label': 'Calculate Correlations', 'category': 'analysis'},
            {'id': 'plot_correlation_network', 'label': 'Plot Correlation Network', 'category': 'analysis'},
            #{'id': 'run_mofa2', 'label': 'Run MOFA2 Analysis', 'category': 'analysis'}
        ]
        
        # Add step numbers and status
        for i, step in enumerate(self.steps):
            step['step'] = i + 1
            step['status'] = 'pending'
    
    def set_current_step(self, step_id):
        """Manually set the current step."""
        step_ids = [step['id'] for step in self.steps]
        if step_id in step_ids:
            self.current_step = step_id
            self._update_status()
        else:
            raise ValueError(f"Unknown step: {step_id}")
    
    def mark_completed(self, step_id):
        """Mark a step as completed."""
        step_ids = [step['id'] for step in self.steps]
        if step_id in step_ids:
            self.completed_steps.add(step_id)
            self._update_status()
    
    def _update_status(self):
        """Update the status of all steps based on current progress."""
        current_idx = next((i for i, step in enumerate(self.steps) if step['id'] == self.current_step), -1)
        
        for i, step in enumerate(self.steps):
            if step['id'] in self.completed_steps:
                step['status'] = 'completed'
            elif i == current_idx and self.current_step:
                step['status'] = 'current'
            elif i < current_idx:
                step['status'] = 'completed'
                self.completed_steps.add(step['id'])
            else:
                step['status'] = 'pending'
    
    def get_progress_stats(self):
        """Get current progress statistics."""
        total_steps = len(self.steps)
        completed_count = len(self.completed_steps)
        current_step = next((step for step in self.steps if step['id'] == self.current_step), None)
        current_label = current_step['label'] if current_step else 'Not started'
        progress_pct = (completed_count / total_steps) * 100
        
        return {
            'completed': completed_count,
            'total': total_steps,
            'percentage': progress_pct,
            'current_step': self.current_step,
            'current_label': current_label
        }
    
    def plot(self, show_plot=True, save_path=None, **kwargs):
        """Plot the current workflow progress as a linear diagram."""
        self._update_status()
        return self._plot_linear(show_plot, save_path, **kwargs)
    
    def _plot_linear(self, show_plot=True, save_path=None, **kwargs):
        """Create linear workflow visualization."""
        # Color mapping for status
        status_colors = {
            'completed': '#2ecc71',    # Green
            'current': '#f39c12',      # Orange
            'pending': '#bdc3c7',      # Light gray
        }
        
        # Calculate positions for linear layout
        x_positions = list(range(len(self.steps)))
        y_position = 0
        
        traces = []
        
        # Create connecting lines
        line_x = []
        line_y = []
        for i in range(len(self.steps) - 1):
            current_status = self.steps[i]['status']
            next_status = self.steps[i + 1]['status']
            
            # Color line based on completion
            if current_status in ['completed', 'current'] and next_status in ['completed', 'current']:
                line_color = '#2ecc71'
                line_width = 4
            else:
                line_color = '#bdc3c7'
                line_width = 2
            
            line_x.extend([i, i + 1, None])
            line_y.extend([y_position, y_position, None])
        
        # Add line trace
        traces.append(go.Scatter(
            x=line_x, y=line_y,
            mode='lines',
            line=dict(width=3, color='#bdc3c7'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Create node traces by status
        for status in ['pending', 'completed', 'current']:
            step_x, step_y, step_text, step_numbers = [], [], [], []
            
            for i, step in enumerate(self.steps):
                if step['status'] == status:
                    step_x.append(i)
                    step_y.append(y_position)
                    step_text.append(f"{step['step']}. {step['label']} ({step['category']})")
                    step_numbers.append(str(step['step']))
            
            if step_x:
                # Special styling for current step
                if status == 'current':
                    traces.append(go.Scatter(
                        x=step_x, y=step_y,
                        mode='markers+text',
                        text=step_numbers,
                        textposition="middle center",
                        textfont=dict(size=14, color='white', family='Arial Black'),
                        hovertext=step_text,
                        hoverinfo='text',
                        marker=dict(
                            size=35,
                            color=status_colors[status],
                            line=dict(width=2, color='red'),
                            symbol='circle'
                        ),
                        name=f'{status.title()} Step',
                        showlegend=True
                    ))
                else:
                    traces.append(go.Scatter(
                        x=step_x, y=step_y,
                        mode='markers+text',
                        text=step_numbers,
                        textposition="middle center",
                        textfont=dict(size=12, color='white' if status != 'pending' else 'gray'),
                        hovertext=step_text,
                        hoverinfo='text',
                        marker=dict(
                            size=30,
                            color=status_colors[status],
                            line=dict(width=1, color='black'),
                            symbol='circle'
                        ),
                        name=f'{status.title()} Steps',
                        showlegend=True
                    ))
        
        # Get progress stats
        stats = self.get_progress_stats()
        title_text = f'<sub>Progress: Completed {stats["completed"]}/{stats["total"]} steps ({stats["percentage"]:.1f}%)<br>Current: {stats["current_label"]}</sub>'
        #title_text = f'<sub>Progress: Completed {stats["completed"]}/{stats["total"]} steps ({stats["percentage"]:.1f}%)</sub>'
        
        # Create figure
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(text=title_text, x=0, font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=60, l=40, r=40, t=100),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=[-0.5, len(self.steps) - 0.5]
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=[-1, 1]
                ),
                plot_bgcolor='white',
                height=200,
                width=1000,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        if show_plot:
            fig.show()
        
        return fig

class Project:
    """Project configuration and directory management with hash-based tagging."""

    def __init__(self, data_processing_hash: str = None, analysis_hash: str = None, overwrite: bool = False):
        self.workflow_tracker = WorkflowProgressTracker()
        self.workflow_tracker.set_current_step('init_project')
        log.info("Initializing Project")
        
        # Handle default config directory
        config_dir = None
        self.default_config_dir = "/home/jovyan/work/input_data/config"
        self.custom_config_dir = "/home/jovyan/work/output_data/*/configs"
        
        if data_processing_hash is None and analysis_hash is None:
            if os.path.isdir(self.default_config_dir):
                config_dir = self.default_config_dir
            else:
                log.error("No configuration directory specified or default path is invalid.")
                raise FileNotFoundError("Configuration directory not found.")
        elif data_processing_hash is not None and analysis_hash is not None:
            matching_dirs = glob.glob(self.custom_config_dir)
            if matching_dirs and os.path.isdir(matching_dirs[0]):
                config_dir = matching_dirs[0]
            else:
                log.error("No configuration directory found for the specified hashes.")
                raise FileNotFoundError("Configuration directory not found for specified hashes.")
        elif data_processing_hash is None and analysis_hash is not None:
            log.error("Both data processing hash and analysis hash must be provided together.")
            raise ValueError("Both data processing hash and analysis hash are required.")
        elif data_processing_hash is not None and analysis_hash is None:
            log.error("Both data processing hash and analysis hash must be provided together.")
            raise ValueError("Both data processing hash and analysis hash are required.")
        
        # Initialize config manager
        self.config_manager = ConfigManager(config_dir, data_processing_hash, analysis_hash)
        self.config, self.data_processing_hash, self.analysis_hash = self.config_manager.load_configs()
        
        # Log configuration info
        if data_processing_hash and analysis_hash:
            log.info(f"Loading existing configuration:")
            log.info(f"  Requested data processing hash: {data_processing_hash}")
            log.info(f"  Requested analysis hash: {analysis_hash}")
        else:
            log.info(f"Using current configuration:")
        
        log.info(f"  Data processing tag: {self.data_processing_hash}")
        log.info(f"  Analysis tag: {self.analysis_hash}")
        
        # Set up project attributes
        self.overwrite = overwrite
        self.project_config = self.config['project']
        self.user_settings = self.config['user_settings']
        self.PI_name = self.project_config['PI_name']
        self.proposal_ID = self.project_config['proposal_ID']
        self.data_types = self.project_config['dataset_list']
        self.study_variables = self.user_settings['variable_list']
        self.project_name = self.user_settings['project_name']
        self.output_dir = self.project_config['results_path']
        self.raw_data_dir = self.project_config['raw_data_path']
        self.project_dir = f"{self.output_dir}/{self.project_name}"
        os.makedirs(self.project_dir, exist_ok=True)
        log.info(f"Project directory: {self.project_dir}")
        self._validate_directory_structure()
        self._complete_tracking('init_project')

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    def save_persistent_config_and_notebook(self):
        """Save the current configuration and notebook with timestamp and tags for this run."""
        
        try:
            # Use hash-based tags for naming
            base_filename = f"Dataset_Processing--{self.data_processing_hash}_Analysis--{self.analysis_hash}"
            
            # Setup directories
            config_dir = os.path.join(self.project_dir, "configs")
            notebooks_dir = os.path.join(self.project_dir, "notebooks")
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(notebooks_dir, exist_ok=True)
            
            # Save all three config files with hash-based naming
            config_files = {
                'project': self.config_manager.project_config_file,
                'data_processing': self.config_manager.data_processing_config_file,
                'analysis': self.config_manager.analysis_config_file
            }
            
            for config_type, source_path in config_files.items():
                dest_filename = f"{base_filename}_{config_type}_config.yml"
                dest_path = os.path.join(config_dir, dest_filename)
                
                if os.path.exists(dest_path):
                    log.warning(f"Configuration file already exists at {dest_path}. It will be updated.")
                
                # Copy config file to timestamped location
                shutil.copy2(source_path, dest_path)
                log.info(f"Configuration saved to: {dest_path}")

            # Handle notebook saving (existing logic)
            this_notebook_path = None
            ipython = get_ipython()
            if ipython and hasattr(ipython, 'kernel'):
                connection_file = ipython.kernel.config['IPKernelApp']['connection_file']
                kernel_id = connection_file.split('-', 1)[1].split('.')[0]
                possible_paths = [
                    f"/notebooks/*.ipynb",
                    f"/work/*.ipynb", 
                    f"*.ipynb",
                    f"/home/jovyan/*.ipynb"
                ]
                for pattern in possible_paths:
                    notebooks = glob.glob(pattern)
                    if notebooks:
                        this_notebook_path = max(notebooks, key=os.path.getmtime)
                        log.info(f"Notebook path identified: {this_notebook_path}")
                        break

            if this_notebook_path and os.path.exists(this_notebook_path):
                notebook_filename = f"{base_filename}_notebook.ipynb"
                new_notebook_path = os.path.join(notebooks_dir, notebook_filename)
                if os.path.exists(new_notebook_path):
                    log.warning(f"Notebook file already exists at {new_notebook_path}. It will be updated.")
                
                # Wait for notebook to save and copy
                start_md5 = hashlib.md5(open(this_notebook_path,'rb').read()).hexdigest()
                current_md5 = start_md5
                max_wait_time = 10
                wait_time = 0
                while start_md5 == current_md5 and wait_time < max_wait_time:
                    time.sleep(2)
                    wait_time += 2
                    if os.path.exists(this_notebook_path):
                        current_md5 = hashlib.md5(open(this_notebook_path,'rb').read()).hexdigest()

                shutil.copy2(this_notebook_path, new_notebook_path)
                log.info(f"Notebook saved to: {new_notebook_path}")
            else:
                log.warning("Could not locate notebook file. Only configuration was saved.")
                log.info("To manually save the notebook, copy your .ipynb file to a persistent location such as:")
                log.info(f"  {os.path.join(notebooks_dir, f'{base_filename}_notebook.ipynb')}")
        
        except Exception as e:
            log.error(f"Failed to save configuration and notebook: {e}")

    def _validate_directory_structure(self) -> Dict[str, Any]:
        """Validate the overall directory structure for consistency."""
        validation_info = {
            'data_processing_dirs': [],
            'analysis_dirs_by_data_hash': {},
            'duplicate_analysis_hashes': {},
            'issues': [],
            'current_config_valid': True
        }
        
        # Only validate if project directory exists
        if not os.path.exists(self.project_dir):
            log.info("Project directory doesn't exist yet - skipping validation.")
            return validation_info
        
        # Find all data processing directories
        data_processing_pattern = os.path.join(self.project_dir, "Dataset_Processing--*")
        data_dirs = glob.glob(data_processing_pattern)
        
        if not data_dirs:
            log.info("No existing data processing directories found.")
            return validation_info
        
        log.info(f"Validating directory structure with {len(data_dirs)} data processing directories...")
        
        for data_dir in data_dirs:
            try:
                data_hash = os.path.basename(data_dir).split('--')[1]
                validation_info['data_processing_dirs'].append(data_hash)
                
                # Find analysis directories under this data processing directory
                analysis_pattern = os.path.join(data_dir, "Analysis--*")
                analysis_dirs = glob.glob(analysis_pattern)
                
                analysis_hashes = []
                for analysis_dir in analysis_dirs:
                    try:
                        analysis_hash = os.path.basename(analysis_dir).split('--')[1]
                        analysis_hashes.append(analysis_hash)
                        
                        # Track where each analysis hash appears
                        if analysis_hash not in validation_info['duplicate_analysis_hashes']:
                            validation_info['duplicate_analysis_hashes'][analysis_hash] = []
                        validation_info['duplicate_analysis_hashes'][analysis_hash].append(data_hash)
                            
                    except Exception as e:
                        log.warning(f"Could not parse analysis directory {analysis_dir}: {e}")
                        validation_info['issues'].append(f"Invalid analysis directory format: {analysis_dir}")
                
                validation_info['analysis_dirs_by_data_hash'][data_hash] = analysis_hashes
                
            except Exception as e:
                log.warning(f"Could not parse data processing directory {data_dir}: {e}")
                validation_info['issues'].append(f"Invalid data processing directory format: {data_dir}")
        
        # Check for legitimate duplicate analysis hashes (same analysis under different data processing)
        duplicates = {ah: data_hashes for ah, data_hashes in validation_info['duplicate_analysis_hashes'].items() 
                    if len(data_hashes) > 1}
        
        if duplicates:
            log.info("Directory structure analysis:")
            for analysis_hash, data_hashes in duplicates.items():
                if len(data_hashes) > 1:
                    log.info(f"  Analysis hash {analysis_hash} exists under data processing hashes: {data_hashes}")
                    # This is normal behavior, not an issue
        
        # Check current configuration validity
        if hasattr(self, 'data_processing_hash') and hasattr(self, 'analysis_hash'):
            expected_dir = os.path.join(
                self.project_dir,
                f"Dataset_Processing--{self.data_processing_hash}",
                f"Analysis--{self.analysis_hash}"
            )
            
            if not os.path.exists(os.path.dirname(expected_dir)):
                log.info(f"Current data processing directory will be created: Dataset_Processing--{self.data_processing_hash}")
            
            if not os.path.exists(expected_dir):
                log.info(f"Current analysis directory will be created: Analysis--{self.analysis_hash}")
        
        # Report validation results
        if validation_info['issues']:
            log.warning(f"Found {len(validation_info['issues'])} directory structure issues:")
            for issue in validation_info['issues']:
                log.warning(f"  - {issue}")
            validation_info['current_config_valid'] = False
        else:
            log.info("Directory structure validation passed - no issues found.")

        return validation_info

class BaseDataHandler:
    """Base class with common data handling functionality."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._cache = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file, inferring format from extension."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in [".tsv", ".tab", ".txt"]:
            return pd.read_csv(file_path, sep='\\t', index_col=0)
        elif ext == ".csv":
            return pd.read_csv(file_path, sep=',', index_col=0)
        elif ext == ".xlsx":
            return pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def save_data(self, data: pd.DataFrame, output_dir: str, filename: str, indexing: bool = True) -> str:
        """Save DataFrame to disk."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        data.to_csv(file_path, index=indexing)
        return file_path
    
    def check_and_load_attribute(self, attribute_name: str, filename: str, overwrite: bool = False) -> bool:
        """Standard pattern for checking/loading cached attributes."""

        file_path = os.path.join(self.output_dir, filename)
  
        # Check cache first
        if attribute_name in self._cache and not overwrite:
            setattr(self, attribute_name, self._cache[attribute_name])
            if hasattr(self, 'dataset_name'):
                log.info(f"{attribute_name} already loaded in memory for {self.dataset_name}. Using cached attribute.")
            else:
                log.info(f"{attribute_name} already loaded in memory. Using cached attribute.")
            return True
            
        # Check disk
        if os.path.exists(file_path) and not overwrite:
            self.clear_cache(attribute_name)
            setattr(self, attribute_name, getattr(self, attribute_name))
            if hasattr(self, 'dataset_name'):
                log.info(f"{attribute_name} file found on disk for {self.dataset_name}. Loading from file.")
            else:
                log.info(f"{attribute_name} file found on disk. Loading from file.")
            return True
            
        if overwrite:
            if hasattr(self, 'dataset_name'):
                log.info(f"{attribute_name} exists for {self.dataset_name} but overwrite=True. Regenerating...")
            else:
                log.info(f"{attribute_name} exists but overwrite=True. Regenerating...")

        return False
    
    def clear_cache(self, key: str):
        """Clear specific cache entry."""
        if key in self._cache:
            del self._cache[key]


class DataAwareBaseHandler(BaseDataHandler):
    """Enhanced base class with automatic data registry integration."""
    
    def __init__(self, output_dir: str, data_registry: Optional['DataRegistry'] = None):
        super().__init__(output_dir)
        self.data_registry = data_registry
        self._auto_register_enabled = data_registry is not None
    
    def _set_df(self, key, filename, df):
        """Enhanced setter that also registers with data registry."""
        super()._set_df(key, filename, df)
        
        if self._auto_register_enabled and not df.empty:
            self._register_dataframe(key, df)
    
    def _register_dataframe(self, attribute_name: str, df: pd.DataFrame):
        """Register DataFrame with the data registry."""
        if not self.data_registry:
            return
        
        object_type = "dataset" if hasattr(self, 'dataset_name') else "analysis"
        object_name = getattr(self, 'dataset_name', '')
        table_name = f"{object_type}_{object_name}_{attribute_name}"
        
        # Get description from attribute docstring or config
        description = self._get_attribute_description(attribute_name)
        
        self.data_registry.register_dataframe(
            df=df,
            table_name=table_name,
            object_type=object_type,
            object_name=object_name,
            attribute_name=attribute_name,
            description=description
        )
    
    def _get_attribute_description(self, attribute_name: str) -> str:
        """Get human-readable description for the attribute."""
        descriptions = {
            'raw_data': 'Raw unprocessed data matrix',
            'filtered_data': 'Data after filtering rare features',
            'normalized_data': 'Final processed data ready for analysis',
            'linked_metadata': 'Metadata linked across samples',
            'feature_network_node_table': 'Network nodes with submodule assignments',
            'feature_correlation_table': 'Pairwise feature correlations',
            'integrated_data': 'Combined data from multiple datasets',
            # Add more as needed
        }
        return descriptions.get(attribute_name, f"Data attribute: {attribute_name}")

class Dataset(DataAwareBaseHandler):
    """Simplified Dataset class using hash-based tagging."""

    def __init__(self, dataset_name: str, project: Project, overwrite: bool = False, superuser: bool = False):
        self.project = project
        self.workflow_tracker = self.project.workflow_tracker
        self.workflow_tracker.set_current_step('create_datasets')
        log.info("Initializing Datasets")
        self.dataset_name = dataset_name
        self.datasets_config = self.project.config['datasets']
        self.dataset_config = self.datasets_config[self.dataset_name]
        self.overwrite = self.project.overwrite

        # Use hash-based tag for output directory
        self.data_processing_tag = project.data_processing_hash
        dataset_outdir = self.set_up_dataset_outdir(self.project, self.data_processing_tag,
                                                    self.dataset_config, self.dataset_name,
                                                    overwrite=self.overwrite)
        self.output_dir = dataset_outdir
        self.dataset_raw_dir = os.path.join(self.project.raw_data_dir, self.dataset_config['dataset_dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        log.info(f"Dataset: {self.dataset_name}")
        log.info(f"Processing tag: {self.data_processing_tag}")
        log.info(f"Output directory: {self.output_dir}")
        
        # Initialize with data registry support (will be set later by Analysis)
        super().__init__(self.output_dir, data_registry=None)

        # Set up filename attributes
        self._setup_dataset_filenames()

        # Configuration
        self.normalization_params = self.dataset_config.get('normalization_parameters', {})
        self.superuser = superuser

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    @staticmethod
    def set_up_dataset_outdir(project: Project, data_processing_tag: str, dataset_config: dict, dataset_name: str, overwrite: bool = False) -> str:
        """Check if the Dataset_Processing--[hash] directory already exists."""
        processing_dir = os.path.join(
            project.project_dir,
            f"Dataset_Processing--{data_processing_tag}",
            dataset_config['dataset_dir']
        )
        if os.path.exists(processing_dir):
            log.info(f"Dataset processing directory already exists. Proceeding with output directory as {processing_dir}")
            return processing_dir
        else:
            log.info(f"Set up {dataset_name} dataset output directory: {processing_dir}")
            return processing_dir

    def _setup_dataset_filenames(self):
        """Setup all dataset filename attributes."""
        manual_file_storage = {
            'raw_data': 'raw_data.csv',
            'raw_metadata': 'raw_metadata.csv',
            'linked_data': 'linked_data.csv',
            'linked_metadata': 'linked_metadata.csv',
            'filtered_data': 'filtered_data.csv',
            'devarianced_data': 'devarianced_data.csv',
            'scaled_data': 'scaled_data.csv',
            'normalized_data': 'normalized_data.csv',
            'pca_grid': 'pca_grid.pdf',
            'annotation_table': 'annotation_table.csv'
        }
        for attr, filename in manual_file_storage.items():
            setattr(self, f"_{attr}_filename", filename)
    
    def _create_property(self, attr_name, filename_attr):
        """Create property with getter/setter pattern."""
        def getter(self):
            return self._get_df(attr_name, getattr(self, filename_attr))
        
        def setter(self, df):
            self._set_df(attr_name, getattr(self, filename_attr), df)
        
        return property(getter, setter)
    
    def _get_df(self, key, filename):
        """Get DataFrame from cache or disk."""
        if key not in self._cache:
            file_path = os.path.join(self.output_dir, filename) if filename else None
            if file_path and os.path.exists(file_path):
                self._cache[key] = self.load_data(file_path)
            else:
                self._cache[key] = pd.DataFrame()
        return self._cache[key]
    
    def _set_df(self, key, filename, df):
        """Set DataFrame to cache and disk."""
        if filename:
            self.save_data(df, self.output_dir, filename, indexing=True)
        self._cache[key] = df

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    def filter_data(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.filter_data function."""
        def _filter_method():
            if self.check_and_load_attribute('filtered_data', self._filtered_data_filename, self.overwrite):
                return
            
            params = self.normalization_params.get('filtering', {})
            call_params = {
                'data': self.linked_data,
                'dataset_name': self.dataset_name,
                'data_type': self.datatype,
                'output_filename': self._filtered_data_filename,
                'output_dir': self.output_dir,
                'filter_method': params.get('method', 'minimum'),
                'filter_value': params.get('value', 0)
            }
            call_params.update(kwargs)
            result = hlp.filter_data(**call_params)
            if result.empty:
                log.error(f"Filtering resulted in empty dataset for {self.dataset_name}. Please adjust filtering parameters.")
                sys.exit(1)
            self.filtered_data = result
            log.info(f"Created table: {self._filtered_data_filename}")
            log.info("Created attribute: filtered_data")
        
        if show_progress:
            self.workflow_tracker.set_current_step('filter_dataset_features')
            _filter_method()
            self._complete_tracking('filter_dataset_features')
            return
        else:
            _filter_method()
            return

    def devariance_data(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Remove low-variance features using external helper function with class integration."""
        def _devariance_method():
            if self.check_and_load_attribute('devarianced_data', self._devarianced_data_filename, self.overwrite):
                return
            
            params = self.normalization_params.get('devariancing', {})
            call_params = {
                'data': self.filtered_data,
                'filter_value': params.get('value', 0),
                'dataset_name': self.dataset_name,
                'output_filename': self._devarianced_data_filename,
                'output_dir': self.output_dir,
                'devariance_mode': params.get('method', 'none')
            }
            call_params.update(kwargs)

            result = hlp.devariance_data(**call_params)
            if result.empty:
                log.error(f"Devariancing resulted in empty dataset for {self.dataset_name}. Please adjust devariancing parameters.")
                sys.exit(1)
            self.devarianced_data = result
            log.info(f"Created table: {self._devarianced_data_filename}")
            log.info("Created attribute: devarianced_data")

        if show_progress:
            self.workflow_tracker.set_current_step('devariance_dataset_features')
            _devariance_method()
            self._complete_tracking('devariance_dataset_features')
            return
        else:
            _devariance_method()
            return

    def scale_data(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.scale_data function."""
        def _scale_method():
            if self.check_and_load_attribute('scaled_data', self._scaled_data_filename, self.overwrite):
                return
            
            params = self.normalization_params.get('scaling', {})
            call_params = {
                'df': self.devarianced_data,
                'output_filename': self._scaled_data_filename,
                'output_dir': self.output_dir,
                'dataset_name': self.dataset_name,
                'log2': params.get('log2', True),
                'norm_method': params.get('method', 'modified_zscore')
            }
            call_params.update(kwargs)
            
            result = hlp.scale_data(**call_params)
            if result.empty:
                log.error(f"Scaling resulted in empty dataset for {self.dataset_name}. Please adjust scaling parameters.")
                sys.exit(1)
            self.scaled_data = result
            log.info(f"Created table: {self._scaled_data_filename}")
            log.info("Created attribute: scaled_data")

        if show_progress:
            self.workflow_tracker.set_current_step('scale_dataset_features')
            _scale_method()
            self._complete_tracking('scale_dataset_features')
            return
        else:
            _scale_method()
            return

    def remove_low_replicable_features(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.remove_low_replicable_features function."""
        def _replicate_method():
            if self.check_and_load_attribute('normalized_data', self._normalized_data_filename, self.overwrite):
                return
            
            params = self.normalization_params.get('replicate_handling', {})
            call_params = {
                'data': self.scaled_data,
                'metadata': self.linked_metadata,
                'dataset_name': self.dataset_name,
                'output_filename': self._normalized_data_filename,
                'output_dir': self.output_dir,
                'method': params.get('method', 'variance'),
                'group_col': params.get('group', 'group'),
                'threshold': params.get('value', 0.6)
            }
            call_params.update(kwargs)
            
            result = hlp.remove_low_replicable_features(**call_params)
            if result.empty:
                log.error(f"Replicability filtering resulted in empty dataset for {self.dataset_name}. Please adjust replicability parameters.")
                sys.exit(1)
            self.normalized_data = result
            log.info(f"Created table: {self._normalized_data_filename}")
            log.info("Created attribute: normalized_data")

        if show_progress:
            self.workflow_tracker.set_current_step('replicability_test_dataset_features')
            _replicate_method()
            self._complete_tracking('replicability_test_dataset_features')
            return
        else:
            _replicate_method()
            return

    def plot_pca(self, overwrite: bool = False, analysis_outdir = None, show_plot = True, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class setup + external hlp.plot_pca function."""
        def _pca_method():
            log.info("Plotting individual PCAs and grid")
            plot_subdir = "pca_plots"
            plot_dir = os.path.join(self.output_dir, plot_subdir)
            os.makedirs(plot_dir, exist_ok=True)
            
            call_params = {
                'data': {"linked": self.linked_data, "normalized": self.normalized_data},
                'metadata': self.linked_metadata,
                'metadata_variables': self.project.study_variables,
                'alpha': 0.75,
                'output_dir': plot_dir,
                'output_filename': self._pca_grid_filename,
                'dataset_name': self.dataset_name,
                'show_plot': show_plot
            }
            call_params.update(kwargs)
            hlp.plot_pca(**call_params)

        if show_progress:
            self.workflow_tracker.set_current_step('plot_pca')
            _pca_method()
            self._complete_tracking('plot_pca')
            return
        else:
            _pca_method()
            return

# Set up properties for Dataset class
manual_file_storage = {
    'raw_data': 'raw_data.csv',
    'raw_metadata': 'raw_metadata.csv',
    'linked_data': 'linked_data.csv',
    'linked_metadata': 'linked_metadata.csv',
    'filtered_data': 'filtered_data.csv',
    'devarianced_data': 'devarianced_data.csv',
    'scaled_data': 'scaled_data.csv',
    'normalized_data': 'normalized_data.csv',
    'pca_grid': 'pca_grid.pdf',
    'annotation_table': 'annotation_table.csv'
}
#for attr in config['datasets']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Dataset, attr, Dataset._create_property(None, attr, f'_{attr}_filename'))

class MX(Dataset):
    """Metabolomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False, last: bool = False, superuser: bool = False):
        super().__init__("mx", project, overwrite, superuser)
        self.workflow_tracker = self.project.workflow_tracker
        self.chromatography = self.dataset_config['chromatography']
        self.polarity = self.dataset_config['polarity']
        self.mode = "untargeted" # Currently only untargeted supported, not configurable
        self.datatype = "peak-height" # Currently only peak-height supported, not configurable
        if 'metabolite_fraction' in self.dataset_config:
            self.metabolite_fraction = self.dataset_config['metabolite_fraction']
            log.info(f"Using only metabolite fraction '{self.metabolite_fraction}' for MX dataset.")
        else:
            self.metabolite_fraction = None
        self._get_raw_metadata(overwrite=self.overwrite, show_progress=False, superuser=superuser)
        self._get_raw_data(overwrite=self.overwrite, show_progress=False)
        self._generate_annotation_table(overwrite=self.overwrite, show_progress=False)
        if last:
            self._complete_tracking('create_datasets')

    def _get_raw_data(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_data_method():
            log.info("Getting Raw Data (MX)")
            if self.check_and_load_attribute('raw_data', self._raw_data_filename, self.overwrite):
                log.info(f"\t{self.dataset_name} data file with {self.raw_data.shape[0]} samples and {self.raw_data.shape[1]} features.")
                return

            result = hlp.get_mx_data(
                input_dir=self.dataset_raw_dir,
                output_dir=self.output_dir,
                output_filename=self._raw_data_filename,
                chromatography=self.chromatography,
                polarity=self.polarity,
                datatype=self.datatype,
            )
            if result.empty:
                log.error(f"No data found for MX dataset with chromatography={self.chromatography} and polarity={self.polarity}. Please check your raw data files.")
                sys.exit(1)

            # Normalize raw data sample-wise using median-of-ratios method
            log.info("Normalizing MX raw data using median-of-ratios method")
            scale = 1.0
            feature_col_name = result.columns[0]
            result = result.set_index(feature_col_name)
            min_dataset_value = result[result > 0].min().min()
            mask = (result > min_dataset_value).any(axis=1)
            counts_pos = result.loc[mask]
            # Replace zeros with a small positive value to avoid log(0) warning
            counts_pos = counts_pos.replace(0, np.finfo(float).eps)
            log_counts = np.log(counts_pos)
            geo_means = np.exp(log_counts.mean(axis=1))
            ratios = counts_pos.divide(geo_means, axis=0)
            size_factors = ratios.median(axis=0)
            norm_counts = result.div(size_factors, axis=1) * scale
            norm_counts = norm_counts.reset_index()

            self.raw_data = norm_counts
            log.info(f"\tCreated raw data for MX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.")
            log.info(f"Created table: {self._raw_data_filename}")
            log.info("Created attribute: raw_data")
        
        if show_progress:
            self.workflow_tracker.set_current_step('load_data')
            _get_data_method()
            self._complete_tracking('load_data')
            return
        else:
            _get_data_method()
            return

    def _get_raw_metadata(self, overwrite: bool = False, show_progress: bool = True, superuser: bool = False) -> None:
        def _get_metadata_method():
            log.info("Getting Raw Metadata (MX)")
            if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, self.overwrite):
                log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
                return

            mx_data_pattern = f"{self.dataset_raw_dir}/*{self.chromatography}*/*_{self.datatype}.csv"
            if not glob.glob(mx_data_pattern):
                mx_parent_folder = hlp.find_mx_parent_folder(
                    pid=self.project.proposal_ID,
                    pi_name=self.project.PI_name,
                    mx_dir=self.dataset_raw_dir,
                    polarity=self.polarity,
                    datatype=self.datatype,
                    chromatography=self.chromatography,
                    overwrite=overwrite,
                    superuser=superuser
                )
                if mx_parent_folder:
                    archives = hlp.gather_mx_files(
                        mx_untargeted_remote=mx_parent_folder,
                        mx_dir=self.dataset_raw_dir,
                        polarity=self.polarity,
                        datatype=self.datatype,
                        chromatography=self.chromatography,
                        extract=True,
                        overwrite=overwrite,
                        superuser=superuser
                    )
            result = hlp.get_mx_metadata(
                output_filename=self._raw_metadata_filename,
                output_dir=self.output_dir,
                input_dir=self.dataset_raw_dir,
                chromatography=self.chromatography,
                polarity=self.polarity
            )
            if result.empty:
                log.error(f"No metadata found for MX dataset with chromatography={self.chromatography} and polarity={self.polarity}. Please check your raw data files.")
                sys.exit(1)
            self.raw_metadata = result
            log.info(f"\tCreated raw metadata for MX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
            log.info(f"Created table: {self._raw_metadata_filename}")
            log.info("Created attribute: raw_metadata")

        if show_progress:
            self.workflow_tracker.set_current_step('load_metadata')
            _get_metadata_method()
            self._complete_tracking('load_metadata')
            return
        else:
            _get_metadata_method()
            return

    def _generate_annotation_table(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Generate metabolite ID to annotation mapping table for metabolomics data."""
        def _generate_annotation_method():
            log.info("Generating Metabolite Annotation Mapping")
            if self.check_and_load_attribute('annotation_table', self._annotation_table_filename, self.overwrite):
                log.info(f"\tAnnotation mapping with {self.annotation_table.shape[0]} rows and {self.annotation_table.shape[1]} columns.")
                return
            
            call_params = {
                'raw_data': self.raw_data,
                'dataset_raw_dir': self.dataset_raw_dir,
                'polarity': self.polarity,
                'output_dir': self.output_dir,
                'output_filename': self._annotation_table_filename
            }

            result = hlp.generate_mx_annotation_table(**call_params)
            if result.empty:
                log.error(f"No annotation mapping generated for MX dataset with chromatography={self.chromatography} and polarity={self.polarity}. Please check your raw data files.")
                sys.exit(1)
            self.annotation_table = result
            log.info(f"Created table: {self._annotation_table_filename}")
            log.info("Created attribute: annotation_table")

        if show_progress:
            self.workflow_tracker.set_current_step('generate_annotation_table')
            _generate_annotation_method()
            self._complete_tracking('generate_annotation_table')
            return
        else:
            _generate_annotation_method()
            return

class TX(Dataset):
    """Transcriptomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False, last: bool = False, superuser: bool = False):
        super().__init__("tx", project, overwrite, superuser)
        self.workflow_tracker = self.project.workflow_tracker
        self.index = 1 # Currently only index 1 supported, not configurable
        self.apid = None
        self.genome_type = self.project.config['project']['genome_type']
        self.datatype = "counts" # Currently only counts supported, not configurable
        self._get_raw_metadata(overwrite=self.overwrite, show_progress=False, superuser=superuser)
        self._get_raw_data(overwrite=self.overwrite, show_progress=False)
        self._generate_annotation_table(overwrite=self.overwrite, show_progress=False)
        if last:
            self._complete_tracking('create_datasets')

    def _get_raw_data(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_data_method():
            log.info("Getting Raw Data (TX)")
            if self.check_and_load_attribute('raw_data', self._raw_data_filename, self.overwrite):
                log.info(f"\t{self.dataset_name} data file with {self.raw_data.shape[0]} samples and {self.raw_data.shape[1]} features.")
                return
            
            result = hlp.get_tx_data(
                input_dir=self.dataset_raw_dir,
                output_dir=self.output_dir,
                output_filename=self._raw_data_filename,
                type=self.datatype,
                overwrite=overwrite
            )
            if result.empty:
                log.error(f"No data found for TX dataset. Please check your raw data files.")
            # Normalize raw data sample-wise using median-of-ratios method
            log.info("Normalizing TX raw data using median-of-ratios method")
            scale = 1.0
            feature_col_name = result.columns[0]
            result = result.set_index(feature_col_name)
            min_dataset_value = result[result > 0].min().min()
            mask = (result > min_dataset_value).any(axis=1)
            counts_pos = result.loc[mask]
            counts_pos = counts_pos.replace(0, np.finfo(float).eps)
            log_counts = np.log(counts_pos)
            geo_means = np.exp(log_counts.mean(axis=1))
            ratios = counts_pos.divide(geo_means, axis=0)
            size_factors = ratios.median(axis=0)
            norm_counts = result.div(size_factors, axis=1) * scale
            norm_counts = norm_counts.reset_index()
            norm_counts = result.div(size_factors, axis=1) * scale
            norm_counts = norm_counts.reset_index()

            self.raw_data = norm_counts
            log.info(f"\tCreated raw data for TX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.")
            log.info(f"Created table: {self._raw_data_filename}")
            log.info("Created attribute: raw_data")

        if show_progress:
            self.workflow_tracker.set_current_step('load_data')
            _get_data_method()
            self._complete_tracking('load_data')
            return
        else:
            _get_data_method()
            return

    def _get_raw_metadata(self, overwrite: bool = False, show_progress: bool = True, superuser: bool = False) -> None:
        def _get_metadata_method():
            log.info("Getting Raw Metadata (TX)")
            if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, self.overwrite):
                self.apid = self.raw_metadata['APID'].iloc[0] if 'APID' in self.raw_metadata.columns else None
                log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
                return
            
            tx_files_path = os.path.join(self.dataset_raw_dir, "all_tx_portal_files.txt")
            if not os.path.exists(tx_files_path):
                tx_files = hlp.find_tx_files(
                    pid=self.project.proposal_ID,
                    tx_dir=self.dataset_raw_dir,
                    tx_index=self.dataset_config['index'],
                    overwrite=self.overwrite,
                    superuser=superuser
                )
                self.apid = hlp.gather_tx_files(
                    file_list=tx_files,
                    tx_index=self.dataset_config['index'],
                    tx_dir=self.dataset_raw_dir,
                    overwrite=self.overwrite,
                    superuser=superuser
                )
            else:
                tx_files = pd.read_csv(tx_files_path, sep='\t')
                if 'index' in self.dataset_config:
                    analysis_index = self.dataset_config['index']
                    log.info("Confirm that the data_processing.yml configuration has the correct analysis index selected (the 'ix' column, integer) from all libraries available for this project:")
                    # Move column ix as index for display purposes
                    tx_files_display = tx_files.set_index('ix')
                    display(tx_files_display)
                    # Copy files from the unprocessed directory over to the dataset_raw_dir
                    unprocessed_dir = os.path.join(self.dataset_raw_dir, "unprocessed")
                    selected_libs = tx_files[tx_files['ix'] == self.dataset_config['index']]
                    for _, row in selected_libs.iterrows():
                        apid = row['APID']
                        gff3_file_src = os.path.join(unprocessed_dir, os.path.basename(row['ref_gff']))
                        gff_file_dest = os.path.join(self.dataset_raw_dir, "genes.gff3")
                        counts_file_src = os.path.join(unprocessed_dir, f"tx_counts_data_{analysis_index}_{apid}.csv")
                        counts_file_dest = os.path.join(self.dataset_raw_dir, "counts.csv")
                        kegg_annotation_file_src = os.path.join(unprocessed_dir, os.path.basename(row['ref_protein_kegg']))
                        kegg_annotation_file_dest = os.path.join(self.dataset_raw_dir, "kegg_annotation_table.tsv")
                        metadata_file_src = os.path.join(unprocessed_dir, f"tx_metadata_{analysis_index}_{apid}.csv")
                        metadata_file_dest = os.path.join(self.dataset_raw_dir, "portal_metadata.csv")
                        for src_file, dest_file in [
                            (gff3_file_src, gff_file_dest),
                            (counts_file_src, counts_file_dest),
                            (kegg_annotation_file_src, kegg_annotation_file_dest),
                            (metadata_file_src, metadata_file_dest)
                        ]:
                            shutil.copy2(src_file, dest_file)
                    self.apid = apid
                    log.info(f"Using APID: {self.apid} for TX dataset metadata extraction.")
            if not hasattr(self, 'apid') or self.apid is None:
                apid_file = os.path.join(self.dataset_raw_dir, "apid.txt")
                if os.path.exists(apid_file):
                    with open(apid_file, 'r') as f:
                        self.apid = f.read().strip()
                else:
                    raise ValueError("APID not found. Cannot proceed without JGI APID.")
            result = hlp.get_tx_metadata(
                tx_files=tx_files,
                output_dir=self.dataset_raw_dir,
                proposal_ID=self.project.proposal_ID,
                apid=str(self.apid),
                overwrite=self.overwrite,
                superuser=superuser
            )
            if result.empty:
                log.error(f"No metadata found for TX dataset. Please check your raw data files.")
                sys.exit(1)
            self.raw_metadata = result
            log.info(f"\tCreated raw metadata for TX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
            log.info(f"Created table: {self._raw_metadata_filename}")
            log.info("Created attribute: raw_metadata")

        if show_progress:
            self.workflow_tracker.set_current_step('load_metadata')
            _get_metadata_method()
            self._complete_tracking('load_metadata')
            return
        else:
            _get_metadata_method()
            return

    def _generate_annotation_table(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Generate gene annotation table for transcriptomics data."""
        def _generate_annotation_method():
            log.info("Generating Gene Annotation Table")
            if self.check_and_load_attribute('annotation_table', self._annotation_table_filename, self.overwrite):
                log.info(f"\tAnnotation table with {self.annotation_table.shape[0]} rows and {self.annotation_table.shape[1]} columns.")
                return
            
            call_params = {
                'raw_data': self.raw_data,
                'raw_data_dir': self.dataset_raw_dir,
                'genome_type': self.genome_type,
                'output_dir': self.output_dir,
                'output_filename': self._annotation_table_filename
            }

            result = hlp.generate_tx_annotation_table(**call_params)
            if result.empty:
                log.error(f"No annotation table generated for TX dataset. Please check your raw data files.")
                sys.exit(1)
            self.annotation_table = result
            log.info(f"Created table: {self._annotation_table_filename}")
            log.info("Created attribute: annotation_table")

        if show_progress:
            self.workflow_tracker.set_current_step('generate_annotation_table')
            _generate_annotation_method()
            self._complete_tracking('generate_annotation_table')
            return
        else:
            _generate_annotation_method()
            return

class Analysis(DataAwareBaseHandler):
    """Enhanced Analysis class with hash-based tagging and AI query capabilities."""
    
    def __init__(self, project: Project, datasets: list = None, overwrite: bool = False):
        self.project = project
        self.workflow_tracker = self.project.workflow_tracker
        self.workflow_tracker.set_current_step('create_analysis')
        log.info("Initializing Analysis")
        self.datasets_config = self.project.config['datasets']
        self.analysis_config = self.project.config['analysis']
        self.metadata_link_script = self.project.project_config['metadata_link']
        self.overwrite = self.project.overwrite

        # Use hash-based tags for output directory
        self.data_processing_tag = project.data_processing_hash
        self.analysis_tag = project.analysis_hash
        self.magi_raw_dir = os.path.join(self.project.raw_data_dir, 'magi')
        analysis_outdir = self._set_up_analysis_outdir(self.project, self.data_processing_tag, 
                                                      self.analysis_tag, overwrite=self.overwrite)
        self.output_dir = analysis_outdir
        os.makedirs(self.output_dir, exist_ok=True)

        log.info(f"Analysis object created")
        log.info(f"Data processing tag: {self.data_processing_tag}")
        log.info(f"Analysis tag: {self.analysis_tag}")
        log.info(f"Output directory: {self.output_dir}")

        # Initialize data registry
        self.data_registry = DataRegistry(db_path=os.path.join(self.output_dir, "analysis_data.db"))
        super().__init__(self.output_dir, self.data_registry)
        
        # Initialize AI agent
        api_key = hlp.load_openai_api_key()
        self.ai_agent = AIQueryAgent(self.data_registry, openai_api_key=api_key)

        self._setup_analysis_filenames()

        self.analysis_parameters = self.analysis_config.get('analysis_parameters', {})
        self.datasets = datasets or []
    
        log.info(f"Created analysis with {len(self.datasets)} datasets.")
        for ds in self.datasets:
            log.info(f"\t- {ds.dataset_name} with output directory: {ds.output_dir}")
        
        self._complete_tracking('create_analysis')

    @staticmethod
    def _set_up_analysis_outdir(project: Project, data_processing_tag: str, analysis_tag: str, overwrite: bool = False) -> str:
        """Check if the Analysis output directory already exists."""
        analysis_dir = os.path.join(
            project.project_dir,
            f"Dataset_Processing--{data_processing_tag}",
            f"Analysis--{analysis_tag}"
        )
        if os.path.exists(analysis_dir):
            log.info(f"Analysis directory already exists. Proceeding with output directory as {analysis_dir}")
            return analysis_dir
        else:
            log.info(f"Set up analysis output directory: {analysis_dir}")
            return analysis_dir

    def _setup_analysis_filenames(self):
        """Setup analysis filename attributes."""
        manual_file_storage = {
            'integrated_metadata': 'integrated_metadata.csv',
            'integrated_data': 'integrated_data.csv',
            'feature_annotation_table': 'feature_annotation_table.csv',
            'integrated_data_selected': 'integrated_data_selected.csv',
            'feature_correlation_table': 'feature_correlation_table.csv',
            'magi_results_table': 'magi_results_table.csv',
            'feature_network_graph': 'feature_network_graph.graphml',
            'feature_network_edge_table': 'feature_network_edge_table.csv',
            'feature_network_node_table': 'feature_network_node_table.csv',
            'functional_enrichment_table': 'functional_enrichment_table.csv',
            #'mofa_data': 'mofa_data.csv',
        }
        for attr, filename in manual_file_storage.items():
            setattr(self, f"_{attr}_filename", filename)

    def _create_property(self, attr_name, filename_attr):
        """Create property with getter/setter pattern for Analysis class."""
        def getter(self):
            return self._get_df(attr_name, getattr(self, filename_attr))
        
        def setter(self, df):
            self._set_df(attr_name, getattr(self, filename_attr), df)
        
        return property(getter, setter)

    def _get_df(self, key, filename):
        """Get DataFrame from cache or disk."""
        if key not in self._cache:
            file_path = os.path.join(self.output_dir, filename) if filename else None
            if file_path and os.path.exists(file_path):
                self._cache[key] = self.load_data(file_path)
            else:
                self._cache[key] = pd.DataFrame()
        return self._cache[key]
    
    def _set_df(self, key, filename, df):
        """Set DataFrame to cache and disk."""
        if filename:
            self.save_data(df, self.output_dir, filename, indexing=True)
        self._cache[key] = df

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    def filter_all_datasets(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Apply filtering to all datasets in the analysis."""
        def _filter_method():
            log.info("Filtering Data")
            for ds in self.datasets:
                log.info(f"Filtering {ds.dataset_name} dataset...")
                ds.filter_data(overwrite=self.overwrite, show_progress=False, **kwargs)

        if show_progress:
            self.workflow_tracker.set_current_step('filter_dataset_features')
            _filter_method()
            self._complete_tracking('filter_dataset_features')
            return
        else:
            _filter_method()
            return

    def devariance_all_datasets(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Apply devariancing to all datasets in the analysis."""
        def _devariance_method():
            log.info("Devariancing Data")
            for ds in self.datasets:
                log.info(f"Devariancing {ds.dataset_name} dataset...")
                ds.devariance_data(overwrite=self.overwrite, show_progress=False, **kwargs)

        if show_progress:
            self.workflow_tracker.set_current_step('devariance_dataset_features')
            _devariance_method()
            self._complete_tracking('devariance_dataset_features')
            return
        else:
            _devariance_method()
            return

    def scale_all_datasets(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Apply scaling to all datasets in the analysis."""
        def _scale_method():
            log.info("Scaling Data")
            for ds in self.datasets:
                log.info(f"Scaling {ds.dataset_name} dataset...")
                ds.scale_data(overwrite=self.overwrite, show_progress=False, **kwargs)

        if show_progress:
            self.workflow_tracker.set_current_step('scale_dataset_features')
            _scale_method()
            self._complete_tracking('scale_dataset_features')
            return
        else:
            _scale_method()
            return

    def replicability_test_all_datasets(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Remove low replicable features from all datasets in the analysis."""
        def _replicate_method():
            log.info("Removing Unreplicable Features")
            for ds in self.datasets:
                log.info(f"Removing low replicable features from {ds.dataset_name} dataset...")
                ds.remove_low_replicable_features(overwrite=self.overwrite, show_progress=False, **kwargs)

        if show_progress:
            self.workflow_tracker.set_current_step('replicability_test_dataset_features')
            _replicate_method()
            self._complete_tracking('replicability_test_dataset_features')
            return
        else:
            _replicate_method()
            return

    def plot_pca_all_datasets(self, overwrite: bool = False, show_plot: bool = True, show_progress: bool = True, **kwargs) -> None:
        """Plot PCA for all datasets in the analysis."""
        def _pca_method():
            log.info("Plotting Individual PCAs and Grid")
            for ds in self.datasets:
                log.info(f"Plotting PCA for {ds.dataset_name} dataset...")
                ds.plot_pca(overwrite=self.overwrite, 
                            analysis_outdir=self.output_dir, 
                            show_plot=show_plot,
                            show_progress=False,
                            **kwargs)

        _pca_method()
        return

    def link_metadata(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Hybrid: Class orchestration + external hlp.link_metadata_with_custom_script function."""
        def _link_metadata_method():
            log.info("Linking analysis datasets along shared metadata")

            # Check if datasets already have linked metadata
            datasets_to_process = [
                ds for ds in self.datasets 
                if not ds.check_and_load_attribute('linked_metadata', ds._linked_metadata_filename, self.overwrite)
            ]
            if not datasets_to_process:
                return
            
            # Call external function
            linked_metadata = hlp.link_metadata_with_custom_script(datasets=self.datasets,
                                                                   custom_script_path=self.metadata_link_script)
            
            # Set results back to datasets
            for ds in self.datasets:
                if linked_metadata[ds.dataset_name].empty:
                    log.error(f"Linking metadata resulted in empty table for {ds.dataset_name}. Please check your metadata linking script and input metadata files.")
                    sys.exit(1)
                ds.linked_metadata = linked_metadata[ds.dataset_name]
                log.info(f"Created linked_metadata for {ds.dataset_name} with {ds.linked_metadata.shape[0]} samples and {ds.linked_metadata.shape[1]} metadata fields.")
                log.info(f"Created table: {ds._linked_metadata_filename}")
                log.info("Created attribute: linked_metadata")

        if show_progress:
            self.workflow_tracker.set_current_step('link_metadata')
            _link_metadata_method()
            self._complete_tracking('link_metadata')
            return
        else:
            _link_metadata_method()
            return

    def link_data(self, overlap_only: bool = True, overwrite: bool = False, show_progress: bool = True) -> None:
        """Hybrid: Class orchestration + external hlp.link_data_across_datasets function."""
        def _link_data_method():
            log.info("Linking analysis datasets along shared samples")

            # Check if all datasets already have linked data
            datasets_to_process = [
                ds for ds in self.datasets 
                if not ds.check_and_load_attribute('linked_data', ds._linked_data_filename, self.overwrite)
            ]
            if not datasets_to_process:
                return

            # Ensure that all datasets have linked metadata
            for ds in self.datasets:
                if not hasattr(ds, 'linked_metadata') or ds.linked_metadata.empty:
                    raise ValueError(f"Dataset {ds.dataset_name} lacks linked_metadata. Cannot proceed with data linking.")

            # Call external function
            linked_data = hlp.link_data_across_datasets(datasets=self.datasets,
                                                           overlap_only=overlap_only)

            # Set results back to datasets
            for ds in self.datasets:
                if linked_data[ds.dataset_name].empty:
                    log.error(f"Linking data resulted in empty table for {ds.dataset_name}. Please check your datasets and linked metadata.")
                    sys.exit(1)
                ds.linked_data = linked_data[ds.dataset_name]
                log.info(f"Created linked_data for {ds.dataset_name} with {ds.linked_data.shape[1]} samples and {ds.linked_data.shape[0]} features.")
                log.info(f"Created table: {ds._linked_data_filename}")
                log.info("Created attribute: linked_data")

        if show_progress:
            self.workflow_tracker.set_current_step('link_data')
            _link_data_method()
            self._complete_tracking('link_data')
            return
        else:
            _link_data_method()
            return

    def plot_dataset_distributions(self, datatype: str = "normalized", show_progress: bool = True) -> None:
        """Plot histograms of feature values for each dataset in the analysis."""
        def _plot_distributions_method():
            log.info("Plotting feature value distributions for all datasets")
            if datatype == "normalized":
                dataframes = {ds.dataset_name: ds.normalized_data for ds in self.datasets if hasattr(ds, "normalized_data")}
            elif datatype == "nonnormalized":
                dataframes = {ds.dataset_name: ds.linked_data for ds in self.datasets if hasattr(ds, "linked_data")}

            hlp.plot_data_variance_histogram(
                dataframes=dataframes,
                datatype=datatype,
                output_dir=self.output_dir,
            )

        _plot_distributions_method()
        return

    def integrate_metadata(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Hybrid: Class validation + external hlp.integrate_metadata function."""
        def _integrate_metadata_method():
            log.info("Integrating metadata across data types")
            if self.check_and_load_attribute('integrated_metadata', self._integrated_metadata_filename, self.overwrite):
                log.info(f"\tIntegrated metadata object 'integrated_metadata' with {self.integrated_metadata.shape[0]} samples and {self.integrated_metadata.shape[1]} metadata fields.")
                return
            
            result = hlp.integrate_metadata(
                datasets=self.datasets,
                metadata_vars=self.project.study_variables,
                unifying_col='unique_group',
                output_filename=self._integrated_metadata_filename,
                output_dir=self.output_dir
            )
            if result.empty:
                log.error(f"Integrating metadata resulted in empty table. Please check your datasets and linked metadata.")
                sys.exit(1)
            self.integrated_metadata = result
            log.info(f"Created a single integrated metadata table with {self.integrated_metadata.shape[0]} samples and {self.integrated_metadata.shape[1]} metadata fields.")
            log.info(f"Created table: {self._integrated_metadata_filename}")
            log.info("Created attribute: integrated_metadata")

        if show_progress:
            self.workflow_tracker.set_current_step('integrate_metadata')
            _integrate_metadata_method()
            self._complete_tracking('integrate_metadata')
            return
        else:
            _integrate_metadata_method()
            return

    def integrate_data(self, overlap_only: bool = True, overwrite: bool = False, show_progress: bool = True) -> None:
        """Hybrid: Class validation + external hlp.integrate_data function."""
        def _integrate_data_method():
            log.info("Integrating data matrices across data types")
            if self.check_and_load_attribute('integrated_data', self._integrated_data_filename, self.overwrite):
                log.info(f"\tIntegrated data object 'integrated_data' with {self.integrated_data.shape[0]} features and {self.integrated_data.shape[1]} samples.")
                return
            
            result = hlp.integrate_data(
                datasets=self.datasets,
                overlap_only=overlap_only,
                output_filename=self._integrated_data_filename,
                output_dir=self.output_dir
            )
            if result.empty:
                log.error(f"Integrating data resulted in empty table. Please check your datasets and linked data.")
                sys.exit(1)
            self.integrated_data = result
            log.info(f"Created a single integrated data table with {self.integrated_data.shape[0]} samples and {self.integrated_data.shape[1]} features.")
            log.info(f"Created table: {self._integrated_data_filename}")
            log.info("Created attribute: integrated_data")

        if show_progress:
            self.workflow_tracker.set_current_step('integrate_data')
            _integrate_data_method()
            self._complete_tracking('integrate_data')
            return
        else:
            _integrate_data_method()
            return

    def annotate_integrated_features(self, overlap_only: bool = True, overwrite: bool = False, show_progress: bool = False) -> pd.DataFrame:       
        """Hybrid: Class orchestration + external annotate_integrated_features function."""
        def _annotate_integrated_features_method():
            log.info("Annotating integrated features")
            if self.check_and_load_attribute('feature_annotation_table', self._feature_annotation_table_filename, self.overwrite):
                log.info(f"\tAnnotated features object 'feature_annotation_table' with {self.feature_annotation_table.shape[0]} features and {self.feature_annotation_table.shape[1]} samples.")
                return
            
            annotation_df = hlp.annotate_integrated_features(
                integrated_data=self.integrated_data,
                datasets=self.datasets,
                output_dir=self.output_dir,
                output_filename=self._feature_annotation_table_filename,
            )

            if annotation_df.empty:
                log.error(f"Annotating integrated features resulted in empty table. Please check your datasets and annotation maps.")
                sys.exit(1)
            self.feature_annotation_table = annotation_df
            log.info(f"Created an annotated integrated features table with {self.feature_annotation_table.shape[0]} entries ({len(self.feature_annotation_table['feature_id'].unique())} unique features) and {self.feature_annotation_table.shape[1]} samples.")
            log.info(f"Created table: {self._feature_annotation_table_filename}")
            log.info("Created attribute: feature_annotation_table")

        if show_progress:
            self.workflow_tracker.set_current_step('annotate_integrated_features')
            _annotate_integrated_features_method()
            self._complete_tracking('annotate_integrated_features')
            return
        else:
            _annotate_integrated_features_method()
            return

    def plot_individual_feature(self, feature_id: str, metadata_cat: str = 'group', save_plot: bool = True) -> None:
        """Plot individual feature abundance by metadata."""
        hlp.plot_feature_abundance_by_metadata(
            data=self.integrated_data,
            metadata=self.integrated_metadata,
            feature=feature_id,
            metadata_group=metadata_cat,
            output_dir=self.output_dir,
            save_plot=save_plot
        )

    def plot_submodule_avg_abundance(self, submodule_name: str, metadata_cat: str = 'group', save_plot: bool = True) -> None:
        """Plot average abundance of features in submodules across metadata groups."""
        hlp.plot_submodule_abundance_by_metadata(
            data=self.integrated_data,
            metadata=self.integrated_metadata,
            node_table=self.feature_network_node_table,
            submodule_name=submodule_name,
            metadata_group=metadata_cat,
            output_dir=self.output_dir,
            save_plot=save_plot
        )

    def perform_feature_selection(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.perform_feature_selection function."""
        def _feature_selection_method():
            log.info("Subsetting Features before Network Analysis")
            if self.check_and_load_attribute('integrated_data_selected', self._integrated_data_selected_filename, self.overwrite):
                log.info(f"\tFeature selection data object 'integrated_data_selected' with {self.integrated_data_selected.shape[0]} features and {self.integrated_data_selected.shape[1]} samples.")
                return
            
            feature_selection_params = self.analysis_parameters.get('feature_selection', {})
            call_params = {
                'data': self.integrated_data,
                'metadata': self.integrated_metadata,
                'config': feature_selection_params, 
                'output_dir': self.output_dir,
                'output_filename': self._integrated_data_selected_filename,
            }
            call_params.update(kwargs)

            result = hlp.perform_feature_selection(**call_params)

            if result.empty:
                log.error(f"Feature selection resulted in empty table. Please check your integrated data and feature selection parameters.")
                sys.exit(1)
            self.integrated_data_selected = result
            log.info(f"Created a subset of the integrated data with {self.integrated_data_selected.shape[0]} samples and {self.integrated_data_selected.shape[1]} features for network analysis.")
            log.info(f"Created table: {self._integrated_data_selected_filename}")
            log.info("Created attribute: integrated_data_selected")

        if show_progress:
            self.workflow_tracker.set_current_step('feature_selection')
            _feature_selection_method()
            self._complete_tracking('feature_selection')
            return
        else:
            _feature_selection_method()
            return

    def run_full_network_analyzer(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        
        correlation_params = self.analysis_parameters.get('correlation', {})
        networking_params = self.analysis_parameters.get('networking', {})
        output_dir = os.path.join(self.output_dir, "network_analyzer_results")
        results = hlp.compare_network_topologies(
            integrated_data=self.integrated_data_selected,
            feature_prefixes=[ds.dataset_name + "_" for ds in self.datasets],
            correlation_params=correlation_params,
            network_params=networking_params,
            annotation_input=self.feature_annotation_table,
            output_dir=output_dir,
            overwrite=overwrite,
            plot_interactive=networking_params.get('interactive_plot', False),
        )

        return results

    def calculate_correlated_features(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.calculate_correlated_features function."""
        def _correlation_method():
            log.info("Calculating Correlated Features")
            if self.check_and_load_attribute('feature_correlation_table', self._feature_correlation_table_filename, self.overwrite):
                log.info(f"\tFeature correlation table object 'feature_correlation_table' with {self.feature_correlation_table.shape[0]} feature pairs.")
                return
            
            correlation_params = self.analysis_parameters.get('correlation', {})
            call_params = {
                'data': self.integrated_data_selected,
                'output_filename': self._feature_correlation_table_filename,
                'output_dir': self.output_dir,
                'output_filename': self._feature_correlation_table_filename,
                'feature_prefixes': [ds.dataset_name + "_" for ds in self.datasets],
                'method': correlation_params.get('corr_method', 'pearson'),
                'cutoff': correlation_params.get('corr_cutoff', 0.5),
                'keep_negative': correlation_params.get('keep_negative', False),
                'block_size': correlation_params.get('block_size', 500),
                'n_jobs': correlation_params.get('cores', -1),
                'corr_mode': correlation_params.get('corr_mode', 'bipartite')
            }
            call_params.update(kwargs)
            
            result = hlp.calculate_correlated_features(**call_params)

            if result.empty:
                log.error(f"Calculating correlated features resulted in empty table. Please check your integrated data and correlation parameters.")
                sys.exit(1)
            self.feature_correlation_table = result
            log.info(f"Created a feature correlation table with {self.feature_correlation_table.shape[0]} feature pairs.")
            log.info(f"Created table: {self._feature_correlation_table_filename}")
            log.info("Created attribute: feature_correlation_table")

        if show_progress:
            self.workflow_tracker.set_current_step('calculate_correlations')
            _correlation_method()
            self._complete_tracking('calculate_correlations')
            return
        else:
            _correlation_method()
            return

    def plot_correlation_network(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        def _network_method():
            log.info("Plotting Correlation Network")
            submodule_subdir = "submodules"
            submodule_dir = os.path.join(self.output_dir, submodule_subdir)
            os.makedirs(submodule_dir, exist_ok=True)
            
            if self.check_and_load_attribute('feature_network_node_table', self._feature_network_node_table_filename, self.overwrite) and \
                self.check_and_load_attribute('feature_network_edge_table', self._feature_network_edge_table_filename, self.overwrite):
                networking_params = self.analysis_parameters.get('networking', {})
                log.info("Displaying existing network visualization...")
                hlp.display_existing_network(
                    graph_file=self._feature_network_graph_filename,
                    node_table=self.feature_network_node_table,
                    edge_table=self.feature_network_edge_table,
                    network_layout=networking_params.get('network_layout', None)
                )
                return
            else: # necessary to clear carryover from previous analyses
                hlp.clear_directory(submodule_dir)

            networking_params = self.analysis_parameters.get('networking', {})
            output_filenames = {
                'graph': self._feature_network_graph_filename,
                'node_table': self._feature_network_node_table_filename,
                'edge_table': self._feature_network_edge_table_filename,
                'submodule_path': submodule_dir
            }

            call_params = {
                'corr_table': self.feature_correlation_table,
                'datasets': self.datasets,
                'integrated_data': self.integrated_data_selected,
                'integrated_metadata': self.integrated_metadata,
                'output_dir': self.output_dir,
                'output_filenames': output_filenames,
                'annotation_df': self.feature_annotation_table,
                'submodule_mode': networking_params.get('submodule_mode', 'community'),
                'network_layout': networking_params.get('network_layout', None),
            }
            call_params.update(kwargs)
            
            node_table, edge_table = hlp.plot_correlation_network(**call_params)

            if node_table.empty or edge_table.empty:
                log.error(f"Plotting correlation network resulted in empty node/edge tables. Please check your correlation table and networking parameters.")
                sys.exit(1)
            else:
                log.info("Created correlation network graph and associated node/edge tables.")

            self.feature_network_node_table = node_table
            self.feature_network_edge_table = edge_table

            log.info(f"Created table: {self._feature_network_node_table_filename}")
            log.info("Created attribute: feature_network_node_table")
            log.info(f"Created table: {self._feature_network_edge_table_filename}")
            log.info("Created attribute: feature_network_edge_table")

        if show_progress:
            self.workflow_tracker.set_current_step('plot_correlation_network')
            _network_method()
            self._complete_tracking('plot_correlation_network')
            return
        else:
            _network_method()
            return

    def perform_functional_enrichment(self, overwrite: bool = False, show_progress: bool = False, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.perform_functional_enrichment function."""
        def _enrichment_method():
            log.info("Performing functional enrichment test on network submodules")
            if self.check_and_load_attribute('functional_enrichment_table', self._functional_enrichment_table_filename, self.overwrite):
                log.info(f"\tFunctional enrichment table object 'functional_enrichment_table' with {self.functional_enrichment_table.shape[0]} functional categories.")
                return

            networking_params = self.analysis_parameters.get('networking', {})
            if networking_params.get('submodule_mode', 'community') == 'none':
                log.warning("Submodule mode is set to 'none'; skipping functional enrichment analysis.")
                return
            functional_enrichment_params = self.analysis_parameters.get('functional_enrichment', {})
            call_params = {
                'node_table': self.feature_network_node_table,
                'annotation_column': functional_enrichment_params.get('annotation_column', None),
                'p_value_threshold': functional_enrichment_params.get('pvalue_cutoff', 0.05),
                'correction_method': functional_enrichment_params.get('correction_method', 'fdr_bh'),
                'min_annotation_count': functional_enrichment_params.get('min_genes_per_term', 1),
                'output_dir': self.output_dir,
                'output_filename': self._functional_enrichment_table_filename,
            }
            call_params.update(kwargs)
            
            enrichment_table = hlp.perform_functional_enrichment(**call_params)

            if enrichment_table.empty:
                log.error(f"Performing functional enrichment resulted in empty table. Please check your node table and enrichment parameters.")
                sys.exit(1)
            self.functional_enrichment_table = enrichment_table
            log.info(f"Created table: {self._functional_enrichment_table_filename}")
            log.info("Created attribute: functional_enrichment_table")

        if show_progress:
            self.workflow_tracker.set_current_step('functional_enrichment')
            _enrichment_method()
            self._complete_tracking('functional_enrichment')
            return
        else:
            _enrichment_method()
            return

    # def run_mofa2_analysis(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
    #     """Hybrid: Class parameter setup + external hlp.run_full_mofa2_analysis function."""
    #     def _mofa_method():
    #         mofa_subdir = "mofa"
    #         mofa_dir = os.path.join(self.output_dir, mofa_subdir)
    #         os.makedirs(mofa_dir, exist_ok=True)
    #         log.info("Running MOFA2 Analysis")
            
    #         # Check with subdirectory path
    #         mofa_data_path = os.path.join(mofa_subdir, self._mofa_data_filename)
    #         if self.check_and_load_attribute('mofa_data', mofa_data_path, self.overwrite):
    #             log.info(f"MOFA2 model already exists. Using existing data.")
    #             return

    #         mofa2_params = self.analysis_parameters.get('mofa', {})
    #         call_params = {
    #             'integrated_data': self.integrated_data_selected,
    #             'mofa2_views': [ds.dataset_name for ds in self.datasets],
    #             'metadata': self.integrated_metadata,
    #             'output_dir': mofa_dir,
    #             'output_filename': self._mofa_data_filename,
    #             'num_factors': mofa2_params.get('num_mofa_factors', 5),
    #             'num_features': 10,
    #             'num_iterations': mofa2_params.get('num_mofa_iterations', 100),
    #             'training_seed': mofa2_params.get('seed_for_training', 555),
    #             'overwrite': self.overwrite
    #         }
    #         call_params.update(kwargs)
            
    #         mofa_data = hlp.run_full_mofa2_analysis(**call_params)

    #         if mofa_data.empty:
    #             log.error(f"Running MOFA2 analysis resulted in empty model. Please check your integrated data and MOFA2 parameters.")
    #             sys.exit(1)
            
    #         # Temporarily update filename to include subdirectory
    #         original_filename = self._mofa_data_filename
    #         self._mofa_data_filename = mofa_data_path
            
    #         # Set the attribute (saves to subdirectory)
    #         self.mofa_data = mofa_data
            
    #         # Restore original filename
    #         self._mofa_data_filename = original_filename

    #         log.info(f"Created table: {self._mofa_data_filename}")
    #         log.info("Created attribute: mofa_data")

    #     if show_progress:
    #         self.workflow_tracker.set_current_step('run_mofa2')
    #         _mofa_method()
    #         self._complete_tracking('run_mofa2')
    #         return
    #     else:
    #         _mofa_method()
    #         return

    def query(self, text_query: str, limit: int = 500) -> pd.DataFrame:
        """Natural language interface to query analysis data."""
        return self.ai_agent.query(text_query, limit)
    
    def register_all_existing_data(self):
        """Register all existing DataFrames with the registry."""
        # Register analysis data
        log.info("Registering data for the analysis object:")
        for attr_name in ['integrated_data', 'integrated_metadata', 'feature_network_node_table', 
                         'feature_correlation_table', 'feature_annotation_table', 'functional_enrichment_table']:
            if hasattr(self, attr_name):
                df = getattr(self, attr_name)
                if not df.empty:
                    log.info(f"Registering analysis attribute '{attr_name}'...")
                    self._register_dataframe(attr_name, df)
        
        # Register dataset data
        for dataset in self.datasets:
            log.info(f"Registering data for the dataset object {dataset.dataset_name}:")
            # Enable data registry for datasets
            dataset.data_registry = self.data_registry
            dataset._auto_register_enabled = True
            
            # Register dataset attributes
            for attr_name in ['raw_data', 'normalized_data', 'linked_metadata', 'linked_data', 
                             'filtered_data', 'scaled_data', 'annotation_table']:
                if hasattr(dataset, attr_name):
                    df = getattr(dataset, attr_name)
                    if not df.empty:
                        # Check if method exists before calling
                        if hasattr(dataset, '_register_dataframe'):
                            log.info(f"Registering dataset attribute '{attr_name}'...")
                            dataset._register_dataframe(attr_name, df)
                        else:
                            # Fallback: register directly through analysis
                            self._register_dataset_dataframe(dataset, attr_name, df)
        
        log.info("Completed registering all existing data.")
    
    def _register_dataset_dataframe(self, dataset, attribute_name: str, df: pd.DataFrame):
        """Helper method to register dataset DataFrames when dataset doesn't have the method."""
        if not self.data_registry or df.empty:
            return
        
        table_name = f"dataset_{dataset.dataset_name}_{attribute_name}"
        description = self._get_attribute_description(attribute_name)
        
        self.data_registry.register_dataframe(
            df=df,
            table_name=table_name,
            object_type="dataset",
            object_name=dataset.dataset_name,
            attribute_name=attribute_name,
            description=description
        )

# Create properties for Analysis class
manual_file_storage = {
    'integrated_metadata': 'integrated_metadata.csv',
    'integrated_data': 'integrated_data.csv',
    'feature_annotation_table': 'feature_annotation_table.csv',
    'integrated_data_selected': 'integrated_data_selected.csv',
    'feature_correlation_table': 'feature_correlation_table.csv',
    'magi_results_table': 'magi_results_table.csv',
    'feature_network_graph': 'feature_network_graph.graphml',
    'feature_network_edge_table': 'feature_network_edge_table.csv',
    'feature_network_node_table': 'feature_network_node_table.csv',
    'functional_enrichment_table': 'functional_enrichment_table.csv',
    #'mofa_data': 'mofa_data.csv'
}
#for attr in config['analysis']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Analysis, attr, Analysis._create_property(None, attr, f'_{attr}_filename'))