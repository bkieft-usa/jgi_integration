import os
import sys
import yaml
import glob
import pandas as pd
import numpy as np
import shutil
from typing import Dict, Any, List, Optional
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

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "\033[47m%(levelname)s - %(message)s\033[0m"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

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
            {'id': 'run_mofa2', 'label': 'Run MOFA2 Analysis', 'category': 'analysis'}
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
    """Project configuration and directory management."""

    def __init__(self, config):
        self.workflow_tracker = WorkflowProgressTracker()
        self.workflow_tracker.set_current_step('init_project')
        log.info("Initializing Project")
        self.config = config
        self.project_config = config['project']
        self.user_settings = config['user_settings']
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
        self._complete_tracking('init_project')

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    def save_persistent_config_and_notebook(self):
        """Save the current configuration and notebook with timestamp and tags for this run."""
        
        try:
            # Get tags for naming
            data_processing_tag = self.config['datasets']['data_processing_tag']
            data_analysis_tag = self.config['analysis']['data_analysis_tag']
            
            # Create base filename pattern
            base_filename = f"Dataset_Processing--{data_processing_tag}_Analysis--{data_analysis_tag}"
            
            # Setup directories
            config_dir = os.path.join(self.project_dir, "configs")
            notebooks_dir = os.path.join(self.project_dir, "notebooks")
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(notebooks_dir, exist_ok=True)
            
            # Define new configuration filename
            config_filename = f"{base_filename}_config.yml"
            config_path = os.path.join(config_dir, config_filename)
            if os.path.exists(config_path):
                log.warning(f"Configuration file already exists at {config_path}. It will be updated.")
            
            # Add metadata to config
            config_with_metadata = self.config.copy()
            config_with_metadata['_metadata'] = {
                'created_at': pd.Timestamp.now().isoformat(),
                'data_processing_tag': data_processing_tag,
                'data_analysis_tag': data_analysis_tag
            }

            # Save config to disk
            with open(config_path, 'w') as f:
                yaml.dump(config_with_metadata, f, default_flow_style=False, sort_keys=False)
            log.info(f"Configuration saved to: {config_path}")

            this_notebook_path = None
            # Try using IPython's get_ipython() for notebook name
            if this_notebook_path is None:
                ipython = get_ipython()
                if ipython and hasattr(ipython, 'kernel'):
                    # Try to get the connection file
                    connection_file = ipython.kernel.config['IPKernelApp']['connection_file']
                    kernel_id = connection_file.split('-', 1)[1].split('.')[0]
                    # Look for notebook files in common locations
                    possible_paths = [
                        f"/notebooks/*.ipynb",
                        f"/work/*.ipynb", 
                        f"*.ipynb",
                        f"/home/jovyan/*.ipynb"
                    ]
                    for pattern in possible_paths:
                        notebooks = glob.glob(pattern)
                        if notebooks:
                            # Use the most recently modified notebook
                            this_notebook_path = max(notebooks, key=os.path.getmtime)
                            log.info(f"Notebook path identified: {this_notebook_path}")
                            break

            if this_notebook_path and os.path.exists(this_notebook_path):
                # Define new notebook filename
                notebook_filename = f"{base_filename}_notebook.ipynb"
                new_notebook_path = os.path.join(notebooks_dir, notebook_filename)
                if os.path.exists(new_notebook_path):
                    log.warning(f"Notebook file already exists at {new_notebook_path}. It will be updated.")
                
                # Get initial hash
                start_md5 = hashlib.md5(open(this_notebook_path,'rb').read()).hexdigest()
                current_md5 = start_md5
                max_wait_time = 30  # seconds
                wait_time = 0
                while start_md5 == current_md5 and wait_time < max_wait_time:
                    time.sleep(2)
                    wait_time += 2
                    if os.path.exists(this_notebook_path):
                        current_md5 = hashlib.md5(open(this_notebook_path,'rb').read()).hexdigest()

                # Copy to output directory
                shutil.copy2(this_notebook_path, new_notebook_path)
                log.info(f"Notebook saved to: {new_notebook_path}")
            else:
                log.warning("Could not locate notebook file. Only configuration was saved.")
                log.info("To manually save the notebook, copy your .ipynb file to a persistent location such as:")
                log.info(f"  {os.path.join(notebooks_dir, f'{base_filename}_notebook.ipynb')}")
        
        except Exception as e:
            log.error(f"Failed to save configuration and notebook: {e}")

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

class Dataset(BaseDataHandler):
    """Simplified Dataset class using hybrid approach."""

    def __init__(self, dataset_name: str, project: Project, overwrite: bool = False):
        self.project = project
        self.workflow_tracker = self.project.workflow_tracker
        self.workflow_tracker.set_current_step('create_datasets')
        log.info("Initializing Datasets")
        self.dataset_name = dataset_name
        self.datasets_config = self.project.config['datasets']
        self.dataset_config = self.datasets_config[self.dataset_name]

        # Set up output directories
        dataset_outdir = self.set_up_dataset_outdir(self.project, self.datasets_config,
                                                    self.dataset_config, self.dataset_name,
                                                    overwrite=overwrite)
        self.output_dir = dataset_outdir
        self.dataset_raw_dir = os.path.join(self.project.raw_data_dir, self.dataset_config['dataset_dir'])
        os.makedirs(self.output_dir, exist_ok=True)
        super().__init__(self.output_dir)

        # Set up filename attributes
        self._setup_dataset_filenames()

        # Configuration
        self.normalization_params = self.dataset_config.get('normalization_parameters', {})
        self.annotation = self.dataset_config.get('annotation', {})

    def _complete_tracking(self, step_id: str):
        """Helper method to track workflow steps with after visualization."""
        # Mark as completed and show updated status (green)
        self.workflow_tracker.mark_completed(step_id)
        self.workflow_tracker.plot(show_plot=True)

    @staticmethod
    def set_up_dataset_outdir(project: Project, datasets_config: dict, dataset_config: dict, dataset_name: str, overwrite: bool = False) -> bool:
        """Check if the Dataset_Processing--[tag] directory already exists."""
        processing_dir = os.path.join(
            project.project_dir,
            f"Dataset_Processing--{datasets_config['data_processing_tag']}",
            dataset_config['dataset_dir']
        )
        if os.path.exists(processing_dir):
            log.info(f"Dataset processing directory already exists. Proceeding with output directory as {processing_dir}")
            return processing_dir
            # if overwrite:
            #     log.info(f"Overwriting existing dataset processing directory for {dataset_name} at {processing_dir}.")
            #     #shutil.rmtree(processing_dir)
            #     return processing_dir
            # else:
            #     error_msg = f"Dataset processing directory already exists for {dataset_name} at {processing_dir}\n" \
            #                 "Please choose a different data processing tag, delete the existing directory, or use the overwrite=True flag before proceeding."
            #     log.error(error_msg)
            #     sys.exit(1)
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
            'annotation_map': 'annotation_map.csv'
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
            if self.check_and_load_attribute('filtered_data', self._filtered_data_filename, overwrite):
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
            if self.check_and_load_attribute('devarianced_data', self._devarianced_data_filename, overwrite):
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
            if self.check_and_load_attribute('scaled_data', self._scaled_data_filename, overwrite):
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
            if self.check_and_load_attribute('normalized_data', self._normalized_data_filename, overwrite):
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
            if os.path.exists(os.path.join(plot_dir, self._pca_grid_filename)) and not overwrite:
                log.info(f"PCA grid plot already exists at {plot_dir}. Skipping.")
                return
            else: # necessary to clear carryover from previous analyses
                hlp.clear_directory(plot_dir)
            
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
    'annotation_map': 'annotation_map.csv'
}
#for attr in config['datasets']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Dataset, attr, Dataset._create_property(None, attr, f'_{attr}_filename'))

class MX(Dataset):
    """Metabolomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False, last: bool = False):
        super().__init__("mx", project, overwrite)
        self.workflow_tracker = self.project.workflow_tracker
        self.chromatography = self.dataset_config['chromatography']
        self.polarity = self.dataset_config['polarity']
        self.mode = "untargeted" # Currently only untargeted supported, not configurable
        self.datatype = "peak-height" # Currently only peak-height supported, not configurable
        self._get_raw_metadata(overwrite=overwrite, show_progress=False)
        self._get_raw_data(overwrite=overwrite, show_progress=False)
        self._generate_annotation_map(overwrite=overwrite, show_progress=False)
        if last:
            self._complete_tracking('create_datasets')

    def _get_raw_data(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_data_method():
            log.info("Getting Raw Data (MX)")
            if self.check_and_load_attribute('raw_data', self._raw_data_filename, overwrite):
                log.info(f"\t{self.dataset_name} data file with {self.raw_data.shape[0]} samples and {self.raw_data.shape[1]} features.")
                return

            result = hlp.get_mx_data(
                input_dir=self.dataset_raw_dir,
                output_dir=self.output_dir,
                output_filename=self._raw_data_filename,
                chromatography=self.chromatography,
                polarity=self.polarity,
                datatype=self.datatype,
                filtered_mx=False,
            )
            if result.empty:
                log.error(f"No data found for MX dataset with chromatography={self.chromatography} and polarity={self.polarity}. Please check your raw data files.")
                sys.exit(1)
            self.raw_data = result
            log.info(f"\tCreated raw data for MX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.\n")
        
        if show_progress:
            self.workflow_tracker.set_current_step('load_data')
            _get_data_method()
            self._complete_tracking('load_data')
            return
        else:
            _get_data_method()
            return

    def _get_raw_metadata(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_metadata_method():
            log.info("Getting Raw Metadata (MX)")
            if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, overwrite):
                log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
                return

            mx_data_pattern = f"{self.dataset_raw_dir}/*{self.chromatography}*/*_{self.datatype}.csv"
            if not glob.glob(mx_data_pattern):
                mx_parent_folder = hlp.find_mx_parent_folder(
                    pid=self.project.proposal_ID,
                    pi_name=self.project.PI_name,
                    script_dir=self.project.script_dir,
                    mx_dir=self.dataset_raw_dir,
                    polarity=self.polarity,
                    datatype=self.datatype,
                    chromatography=self.chromatography,
                    filtered_mx=False,
                    overwrite=overwrite
                )
                if mx_parent_folder:
                    hlp.gather_mx_files(
                        mx_untargeted_remote=mx_parent_folder,
                        script_dir=self.project.script_dir,
                        mx_dir=self.dataset_raw_dir,
                        polarity=self.polarity,
                        datatype=self.datatype,
                        chromatography=self.chromatography,
                        filtered_mx=False,
                        extract=True,
                        overwrite=overwrite
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
            log.info(f"\tCreated raw metadata for MX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.\n")
        
        if show_progress:
            self.workflow_tracker.set_current_step('load_metadata')
            _get_metadata_method()
            self._complete_tracking('load_metadata')
            return
        else:
            _get_metadata_method()
            return

    def _generate_annotation_map(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Generate metabolite ID to annotation mapping table for metabolomics data."""
        def _generate_annotation_method():
            log.info("Generating Metabolite Annotation Mapping")
            if self.check_and_load_attribute('annotation_map', self._annotation_map_filename, overwrite):
                log.info(f"\tAnnotation mapping with {self.annotation_map.shape[0]} rows and {self.annotation_map.shape[1]} columns.")
                return
            
            call_params = {
                'raw_data': self.raw_data,
                'dataset_raw_dir': self.dataset_raw_dir,
                'polarity': self.polarity,
                'output_dir': self.output_dir,
                'output_filename': self._annotation_map_filename
            }

            result = hlp.generate_mx_annotation_map(**call_params)
            if result.empty:
                log.error(f"No annotation mapping generated for MX dataset with chromatography={self.chromatography} and polarity={self.polarity}. Please check your raw data files.")
                sys.exit(1)
            self.annotation_map = result

        if show_progress:
            self.workflow_tracker.set_current_step('generate_annotation_map')
            _generate_annotation_method()
            self._complete_tracking('generate_annotation_map')
            return
        else:
            _generate_annotation_method()
            return

class TX(Dataset):
    """Transcriptomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False, last: bool = False):
        super().__init__("tx", project, overwrite)
        self.workflow_tracker = self.project.workflow_tracker
        self.index = 1 # Currently only index 1 supported, not configurable
        self.apid = None
        self.datatype = "counts" # Currently only counts supported, not configurable
        self._get_raw_metadata(overwrite=overwrite, show_progress=False)
        self._get_raw_data(overwrite=overwrite, show_progress=False)
        self._generate_annotation_map(overwrite=overwrite, show_progress=False)
        if last:
            self._complete_tracking('create_datasets')

    def _get_raw_data(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_data_method():
            log.info("Getting Raw Data (TX)")
            if self.check_and_load_attribute('raw_data', self._raw_data_filename, overwrite):
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
                sys.exit(1)
            self.raw_data = result
            log.info(f"\tCreated raw data for TX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.\n")

        if show_progress:
            self.workflow_tracker.set_current_step('load_data')
            _get_data_method()
            self._complete_tracking('load_data')
            return
        else:
            _get_data_method()
            return

    def _get_raw_metadata(self, overwrite: bool = False, show_progress: bool = True) -> None:
        def _get_metadata_method():
            log.info("Getting Raw Metadata (TX)")
            if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, overwrite):
                self.apid = self.raw_metadata['APID'].iloc[0] if 'APID' in self.raw_metadata.columns else None
                log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
                return
            
            tx_files_path = os.path.join(self.dataset_raw_dir, "all_tx_portal_files.txt")
            if not os.path.exists(tx_files_path):
                tx_files = hlp.find_tx_files(
                    pid=self.project.proposal_ID,
                    script_dir=self.project.script_dir,
                    tx_dir=self.dataset_raw_dir,
                    tx_index=self.dataset_config['index'],
                    overwrite=overwrite
                )
                self.apid = hlp.gather_tx_files(
                    file_list=tx_files,
                    tx_index=self.dataset_config['index'],
                    script_dir=self.project.script_dir,
                    tx_dir=self.dataset_raw_dir,
                    overwrite=overwrite
                )
            else:
                tx_files = pd.read_csv(tx_files_path, sep='\t')
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
                overwrite=overwrite
            )
            if result.empty:
                log.error(f"No metadata found for TX dataset. Please check your raw data files.")
                sys.exit(1)
            self.raw_metadata = result
            log.info(f"\tCreated raw metadata for TX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.\n")

        if show_progress:
            self.workflow_tracker.set_current_step('load_metadata')
            _get_metadata_method()
            self._complete_tracking('load_metadata')
            return
        else:
            _get_metadata_method()
            return

    def _generate_annotation_map(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Generate gene ID to GO annotation mapping table for transcriptomics data."""
        def _generate_annotation_method():
            log.info("Generating Gene-GO Annotation Mapping")
            if self.check_and_load_attribute('annotation_map', self._annotation_map_filename, overwrite):
                log.info(f"\tAnnotation mapping with {self.annotation_map.shape[0]} rows and {self.annotation_map.shape[1]} columns.")
                return
            
            call_params = {
                'raw_data': self.raw_data,
                'dataset_raw_dir': self.dataset_raw_dir,
                'output_dir': self.output_dir,
                'output_filename': self._annotation_map_filename
            }

            result = hlp.generate_tx_annotation_map(**call_params)
            if result.empty:
                log.error(f"No annotation mapping generated for TX dataset. Please check your raw data files.")
                sys.exit(1)
            self.annotation_map = result

        if show_progress:
            self.workflow_tracker.set_current_step('generate_annotation_map')
            _generate_annotation_method()
            self._complete_tracking('generate_annotation_map')
            return
        else:
            _generate_annotation_method()
            return

class Analysis(BaseDataHandler):
    """Simplified Analysis class using consistent hybrid approach."""

    def __init__(self, project: Project, datasets: list = None, overwrite: bool = False):
        self.project = project
        self.workflow_tracker = self.project.workflow_tracker
        self.workflow_tracker.set_current_step('create_analysis')
        log.info("Initializing Analysis")
        self.datasets_config = self.project.config['datasets']
        self.analysis_config = self.project.config['analysis']
        self.metadata_link_script = self.project.project_config['metadata_link']

        # Check if analysis directory exists
        analysis_outdir = self._set_up_analysis_outdir(self.project, self.datasets_config, 
                                                      self.analysis_config, overwrite=overwrite)
        self.output_dir = analysis_outdir
        os.makedirs(self.output_dir, exist_ok=True)
        super().__init__(self.output_dir)

        self._setup_analysis_filenames()

        self.analysis_parameters = self.analysis_config.get('analysis_parameters', {})
        self.datasets = datasets or []
    
        log.info(f"Created analysis with {len(self.datasets)} datasets.")
        for ds in self.datasets:
            log.info(f"\t- {ds.dataset_name} with output directory: {ds.output_dir}")
        
        self._complete_tracking('create_analysis')

    @staticmethod
    def _set_up_analysis_outdir(project: Project, datasets_config: dict, analysis_config: dict, overwrite: bool = False) -> str:
        """Check if the Analysis output directory already exists."""
        analysis_dir = os.path.join(
            project.project_dir,
            f"Dataset_Processing--{datasets_config['data_processing_tag']}",
            f"Analysis--{analysis_config['data_analysis_tag']}"
        )
        if os.path.exists(analysis_dir):
            if overwrite:
                log.info(f"Overwriting existing analysis directory {analysis_dir}.")
                #shutil.rmtree(analysis_dir)
                return analysis_dir
            else:
                error_msg = f"Analysis directory already exists at {analysis_dir}\n" \
                            "Please choose a different analysis tag, delete the existing directory, or use the overwrite=True flag before proceeding."
                log.error(error_msg)
                sys.exit(1)
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
            'feature_network_graph': 'feature_network_graph.graphml',
            'feature_network_edge_table': 'feature_network_edge_table.csv',
            'feature_network_node_table': 'feature_network_node_table.csv',
            'functional_enrichment_table': 'functional_enrichment_table.csv',
            'mofa_model': 'mofa_model.hdf5',
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
            for ds in tqdm(self.datasets, desc=f"Filtering {len(self.datasets)} Datasets", unit="dataset", leave=True, colour="green"):
                log.info(f"Filtering {ds.dataset_name} dataset...")
                ds.filter_data(overwrite=overwrite, show_progress=False, **kwargs)

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
            for ds in tqdm(self.datasets, desc=f"Devariancing {len(self.datasets)} Datasets", unit="dataset", leave=True, colour="green"):
                log.info(f"Devariancing {ds.dataset_name} dataset...")
                ds.devariance_data(overwrite=overwrite, show_progress=False, **kwargs)

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
            for ds in tqdm(self.datasets, desc=f"Scaling {len(self.datasets)} Datasets", unit="dataset", leave=True, colour="green"):
                log.info(f"Scaling {ds.dataset_name} dataset...")
                ds.scale_data(overwrite=overwrite, show_progress=False, **kwargs)

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
            for ds in tqdm(self.datasets, desc=f"Removing low replicable features from {len(self.datasets)} Datasets", unit="dataset", leave=True, colour="green"):
                log.info(f"Removing low replicable features from {ds.dataset_name} dataset...")
                ds.remove_low_replicable_features(overwrite=overwrite, show_progress=False, **kwargs)

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
                ds.plot_pca(overwrite=overwrite, 
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
                if not ds.check_and_load_attribute('linked_metadata', ds._linked_metadata_filename, overwrite)
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
                log.info(f"Created linked_metadata for {ds.dataset_name} with {ds.linked_metadata.shape[0]} samples and {ds.linked_metadata.shape[1]} metadata fields.\n")

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
                if not ds.check_and_load_attribute('linked_data', ds._linked_data_filename, overwrite)
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
                log.info(f"Created linked_data for {ds.dataset_name} with {ds.linked_data.shape[1]} samples and {ds.linked_data.shape[0]} features.\n")

        if show_progress:
            self.workflow_tracker.set_current_step('link_data')
            _link_data_method()
            self._complete_tracking('link_data')
            return
        else:
            _link_data_method()
            return

    def plot_dataset_distributions(self, datatype: str = "normalized", bins: int = 50, transparency: float = 0.8, xlog: bool = False, show_plot: bool = True, show_progress: bool = True) -> None:
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
                bins=bins,
                transparency=transparency,
                xlog=xlog,
                output_dir=self.output_dir,
                show_plot=show_plot
            )

        _plot_distributions_method()
        return

    def integrate_metadata(self, overwrite: bool = False, show_progress: bool = True) -> None:
        """Hybrid: Class validation + external hlp.integrate_metadata function."""
        def _integrate_metadata_method():
            log.info("Integrating metadata across data types")
            if self.check_and_load_attribute('integrated_metadata', self._integrated_metadata_filename, overwrite):
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
            log.info(f"Created a single integrated metadata table with {self.integrated_metadata.shape[0]} samples and {self.integrated_metadata.shape[1]} metadata fields.\n")

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
            if self.check_and_load_attribute('integrated_data', self._integrated_data_filename, overwrite):
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
            log.info(f"Created a single integrated data table with {self.integrated_data.shape[0]} samples and {self.integrated_data.shape[1]} features.\n")

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
            if self.check_and_load_attribute('feature_annotation_table', self._feature_annotation_table_filename, overwrite):
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
            log.info(f"Created an annotated integrated features table with {self.feature_annotation_table.shape[0]} entries ({len(self.feature_annotation_table['feature_id'].unique())} unique features) and {self.feature_annotation_table.shape[1]} samples.\n")

        if show_progress:
            self.workflow_tracker.set_current_step('annotate_integrated_features')
            _annotate_integrated_features_method()
            self._complete_tracking('annotate_integrated_features')
            return
        else:
            _annotate_integrated_features_method()
            return

    def perform_feature_selection(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.perform_feature_selection function."""
        def _feature_selection_method():
            log.info("Subsetting Features before Network Analysis")
            if self.check_and_load_attribute('integrated_data_selected', self._integrated_data_selected_filename, overwrite):
                log.info(f"\tFeature selection data object 'integrated_data_selected' with {self.integrated_data_selected.shape[0]} features and {self.integrated_data_selected.shape[1]} samples.")
                return
            
            feature_selection_params = self.analysis_parameters.get('feature_selection', {})
            call_params = {
                'data': self.integrated_data,
                'metadata': self.integrated_metadata,
                'config': feature_selection_params, 
                'max_features': feature_selection_params.get('max_features', 5000),
                'output_dir': self.output_dir,
                'output_filename': self._integrated_data_selected_filename,
            }
            call_params.update(kwargs)

            result = hlp.perform_feature_selection(**call_params)

            if result.empty:
                log.error(f"Feature selection resulted in empty table. Please check your integrated data and feature selection parameters.")
                sys.exit(1)
            self.integrated_data_selected = result
            log.info(f"Created a subset of the integrated data with {self.integrated_data_selected.shape[0]} samples and {self.integrated_data_selected.shape[1]} features for network analysis.\n")

        if show_progress:
            self.workflow_tracker.set_current_step('feature_selection')
            _feature_selection_method()
            self._complete_tracking('feature_selection')
            return
        else:
            _feature_selection_method()
            return

    def calculate_correlated_features(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.calculate_correlated_features function."""
        def _correlation_method():
            log.info("Calculating Correlated Features")
            if self.check_and_load_attribute('feature_correlation_table', self._feature_correlation_table_filename, overwrite):
                log.info(f"\tFeature correlation table object 'feature_correlation_table' with {self.feature_correlation_table.shape[0]} feature pairs.")
                return
            
            # Get parameters from config with defaults
            correlation_params = self.analysis_parameters.get('correlation', {})
            call_params = {
                'data': self.integrated_data_selected,
                'output_filename': self._feature_correlation_table_filename,
                'output_dir': self.output_dir,
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
            log.info(f"Created a feature correlation table with {self.feature_correlation_table.shape[0]} feature pairs.\n")

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
            network_subdir = "feature_network"
            submodule_subdir = "submodules"
            network_dir = os.path.join(self.output_dir, network_subdir)
            submodule_dir = os.path.join(network_dir, submodule_subdir)
            os.makedirs(network_dir, exist_ok=True)
            os.makedirs(submodule_dir, exist_ok=True)
            if self.check_and_load_attribute('feature_network_node_table', os.path.join(network_subdir, self._feature_network_node_table_filename), overwrite) or \
                self.check_and_load_attribute('feature_network_edge_table', os.path.join(network_subdir, self._feature_network_edge_table_filename), overwrite):
                return
            else: # necessary to clear carryover from previous analyses
                hlp.clear_directory(network_dir)
                hlp.clear_directory(submodule_dir)

            networking_params = self.analysis_parameters.get('networking', {})
            output_filenames = {
                'graph': os.path.join(network_dir, self._feature_network_graph_filename),
                'node_table': os.path.join(network_dir, self._feature_network_node_table_filename),
                'edge_table': os.path.join(network_dir, self._feature_network_edge_table_filename),
                'submodule_path': submodule_dir
            }

            call_params = {
                'corr_table': self.feature_correlation_table,
                'datasets': self.datasets,
                'integrated_data': self.integrated_data_selected,
                'integrated_metadata': self.integrated_metadata,
                'output_filenames': output_filenames,
                'annotation_df': self.feature_annotation_table,
                'submodule_mode': networking_params.get('submodule_mode', 'community'),
                'show_plot': networking_params.get('interactive_plot', False),
                'interactive_layout': networking_params.get('interactive_layout', None),
                'wgcna_params': networking_params.get('wgcna_params', {"beta": 5, "min_module_size": 10, "distance_cutoff": 0.25})
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
            if self.check_and_load_attribute('functional_enrichment_table', self._functional_enrichment_table_filename, overwrite):
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
                'output_dir': os.path.join(self.output_dir, "feature_network"),
                'output_filename': self._functional_enrichment_table_filename,
            }
            call_params.update(kwargs)
            
            enrichment_table = hlp.perform_functional_enrichment(**call_params)

            if enrichment_table.empty:
                log.error(f"Performing functional enrichment resulted in empty table. Please check your node table and enrichment parameters.")
                sys.exit(1)
            self.functional_enrichment_table = enrichment_table

        if show_progress:
            self.workflow_tracker.set_current_step('functional_enrichment')
            _enrichment_method()
            self._complete_tracking('functional_enrichment')
            return
        else:
            _enrichment_method()
            return

    def run_mofa2_analysis(self, overwrite: bool = False, show_progress: bool = True, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.run_full_mofa2_analysis function."""
        def _mofa_method():
            mofa_subdir = "mofa"
            mofa_dir = os.path.join(self.output_dir, mofa_subdir)
            os.makedirs(mofa_dir, exist_ok=True)
            log.info("Running MOFA2 Analysis")
            if os.path.exists(os.path.join(mofa_subdir, self._mofa_model_filename)) and not overwrite:
                log.info(f"MOFA2 model already exists in {mofa_dir}. Skipping.")
                return

            mofa2_params = self.analysis_parameters.get('mofa', {})
            call_params = {
                'integrated_data': self.integrated_data_selected,
                'mofa2_views': [ds.dataset_name for ds in self.datasets],
                'metadata': self.integrated_metadata,
                'output_dir': mofa_dir,
                'output_filename': self._mofa_model_filename,
                'num_factors': mofa2_params.get('num_mofa_factors', 5),
                'num_features': 10,
                'num_iterations': mofa2_params.get('num_mofa_iterations', 100),
                'training_seed': mofa2_params.get('seed_for_training', 555),
                'overwrite': overwrite
            }
            call_params.update(kwargs)
            
            hlp.run_full_mofa2_analysis(**call_params)

        if show_progress:
            self.workflow_tracker.set_current_step('run_mofa2')
            _mofa_method()
            self._complete_tracking('run_mofa2')
            return
        else:
            _mofa_method()
            return

# Create properties for Analysis class
manual_file_storage = {
    'integrated_metadata': 'integrated_metadata.csv',
    'integrated_data': 'integrated_data.csv',
    'feature_annotation_table': 'feature_annotation_table.csv',
    'integrated_data_selected': 'integrated_data_selected.csv',
    'feature_correlation_table': 'feature_correlation_table.csv',
    'feature_network_graph': 'feature_network_graph.graphml',
    'feature_network_edge_table': 'feature_network_edge_table.csv',
    'feature_network_node_table': 'feature_network_node_table.csv',
    'functional_enrichment_table': 'functional_enrichment_table.csv',
    'mofa_model': 'mofa_model.hdf5'
}
#for attr in config['analysis']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Analysis, attr, Analysis._create_property(None, attr, f'_{attr}_filename'))