import os
import sys
import yaml
import glob
import pandas as pd
import numpy as np
import shutil
from typing import Dict, Any, List, Optional, Tuple
from IPython import get_ipython
import tools.helpers as hlp
import logging
import time
import hashlib
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
        self.analysis_config_file = self._detect_analysis_config_file()

        if self.target_data_hash and self.target_analysis_hash:
            self._load_from_persistent_configs()
        elif not self._standard_configs_exist():
            log.warning("Standard config files not found and no hashes specified.")

    def _detect_analysis_config_file(self) -> Path:
        """
        Detect the analysis config file in the config directory.

        Raises ``FileNotFoundError`` if ``analysis.yml`` is not found.
        """
        candidate = self.config_dir / "analysis.yml"
        if candidate.exists():
            log.info(f"Using analysis config: {candidate.name}")
            return candidate
        raise FileNotFoundError(
            f"No analysis config file found in {self.config_dir}. "
            "Expected: analysis.yml"
        )

    def _standard_configs_exist(self) -> bool:
        """Check if standard config files exist."""
        return all([
            self.project_config_file.exists(),
            self.data_processing_config_file.exists(),
            self.analysis_config_file.exists(),
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
            'analysis': self.analysis_config_file,
        }

        found_configs = {}
        for config_file in config_files:
            name_parts = config_file.stem.split('_')
            if len(name_parts) >= 3:
                config_type = name_parts[-2]  # e.g., 'project', 'data', 'analysis'

                # Handle both 'data' and 'data_processing' naming
                if config_type in ('data', 'processing'):
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
        """
        Load all config files and generate hashes.

        ``analysis.yml`` must have a top-level ``analysis:`` key.  The sub-dict
        is extracted so that ``combined_config['analysis']`` contains the flat
        config that ``Analysis.__init__`` reads via ``self.project.config['analysis']``.
        """
        # Load project config
        with open(self.project_config_file, 'r') as f:
            project_config = yaml.safe_load(f)

        # Load data processing config and generate hash
        with open(self.data_processing_config_file, 'r') as f:
            data_processing_config = yaml.safe_load(f)
        data_processing_hash = self._generate_config_hash(data_processing_config)

        # Load analysis config and generate hash
        with open(self.analysis_config_file, 'r') as f:
            raw_analysis_config = yaml.safe_load(f)
        analysis_hash = self._generate_config_hash(raw_analysis_config)

        # Extract the flat 'analysis' sub-dict
        if not (isinstance(raw_analysis_config, dict) and 'analysis' in raw_analysis_config):
            raise ValueError(
                f"analysis.yml must have a top-level 'analysis:' key. "
                f"Check {self.analysis_config_file}."
            )
        analysis_config = raw_analysis_config['analysis']

        # Verify hashes match if they were specified
        if self.target_data_hash and data_processing_hash != self.target_data_hash:
            log.warning(f"Data processing hash mismatch! Expected: {self.target_data_hash}, Got: {data_processing_hash}")

        if self.target_analysis_hash and analysis_hash != self.target_analysis_hash:
            log.warning(f"Analysis hash mismatch! Expected: {self.target_analysis_hash}, Got: {analysis_hash}")

        # Combine all configs; 'analysis' key holds the flat branch config dict
        combined_config = {
            **project_config,
            **data_processing_config,
            'analysis': analysis_config,
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
    
class Project:
    """Project configuration and directory management with hash-based tagging."""

    def __init__(self, data_processing_hash: str = None, analysis_hash: str = None, overwrite: bool = False):
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
        self.cache_dir = self.project_config['results_path'] + "/cache"
        self.raw_data_dir = self.project_config['raw_data_path']
        self.project_dir = f"{self.output_dir}/{self.project_name}"
        os.makedirs(self.project_dir, exist_ok=True)
        log.info(f"Project directory: {self.project_dir}")
        self._validate_directory_structure()

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


class Dataset(BaseDataHandler):
    """Simplified Dataset class using hash-based tagging."""

    def __init__(self, dataset_name: str, project: Project, overwrite: bool = False, superuser: bool = False):
        self.project = project
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
        
        super().__init__(self.output_dir)

        # Set up filename attributes
        self._setup_dataset_filenames()

        # Configuration
        self.normalization_params = self.dataset_config.get('normalization_parameters', {})
        self.superuser = superuser

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
            'replicate_filtered_data': 'replicate_filtered_data.csv',
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

    def filter_data(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.filter_data function."""
        if self.check_and_load_attribute('filtered_data', self._filtered_data_filename, self.overwrite):
            return

        step = self.normalization_params.get('filtering', {})
        p = step.get('params', step)   # new layout: step['params']; old layout: step itself
        call_params = {
            'data': self.linked_data,
            'dataset_name': self.dataset_name,
            'data_type': self.datatype,
            'output_filename': self._filtered_data_filename,
            'output_dir': self.output_dir,
            'filter_method': step.get('method', 'minimum'),
            'filter_value': p.get('value', 0)
        }
        call_params.update(kwargs)
        result = hlp.filter_data(**call_params)
        if result.empty:
            log.error(f"Filtering resulted in empty dataset for {self.dataset_name}. Please adjust filtering parameters.")
            sys.exit(1)
        self.filtered_data = result
        log.info(f"Created table: {self._filtered_data_filename}")
        log.info("Created attribute: filtered_data")

    def devariance_data(self, overwrite: bool = False, **kwargs) -> None:
        """Remove low-variance features using external helper function with class integration."""
        if self.check_and_load_attribute('devarianced_data', self._devarianced_data_filename, self.overwrite):
            return

        step = self.normalization_params.get('devariancing', {})
        p = step.get('params', step)   # new layout: step['params']; old layout: step itself
        call_params = {
            'data': self.filtered_data,
            'filter_value': p.get('value', 0),
            'dataset_name': self.dataset_name,
            'output_filename': self._devarianced_data_filename,
            'output_dir': self.output_dir,
            'devariance_mode': step.get('method', 'none')
        }
        call_params.update(kwargs)

        result = hlp.devariance_data(**call_params)
        if result.empty:
            log.error(f"Devariancing resulted in empty dataset for {self.dataset_name}. Please adjust devariancing parameters.")
            sys.exit(1)
        self.devarianced_data = result
        log.info(f"Created table: {self._devarianced_data_filename}")
        log.info("Created attribute: devarianced_data")

    def scale_data(self, overwrite: bool = False, **kwargs) -> None:
        """
        Per-dataset scaling step for replicate_matched mode.

        Reads ``replicate_filtered_data``, applies the scaling method configured in
        ``data_processing.yml`` under ``normalization_parameters.scaling``, and writes
        the result to ``scaled_data``.

        This step is only meaningful in **replicate_matched** mode.  In **lfc** mode,
        ``Analysis.scale_all_datasets()`` skips this step automatically because LFC
        computation is scale-invariant (ratios cancel absolute differences).

        Supported scaling methods (set in data_processing.yml → scaling.method):
        - ``"vst"``            : arcsinh(sqrt(x)) variance-stabilising transform + z-score
        - ``"zscore"``         : log2(x+1) then z-score per sample
        - ``"modified_zscore"``: log2(x+1) then modified z-score per sample
        - ``"rank_normal"``    : rank-based inverse normal transformation
        - ``"quantile"``       : quantile normalisation (forces identical distributions)
        - ``"none"``           : pass-through (no scaling applied)
        """
        if self.check_and_load_attribute('scaled_data', self._scaled_data_filename, self.overwrite):
            log.info(f"\tScaled data already exists for {self.dataset_name}.")
            return

        step = self.normalization_params.get('scaling', {})
        p = step.get('params', step)   # new layout: step['params']; old layout: step itself
        norm_method = step.get('method', 'vst')
        log2 = True

        log.info(f"Scaling {self.dataset_name} data using method='{norm_method}'...")

        call_params = {
            'df': self.replicate_filtered_data,
            'output_filename': self._scaled_data_filename,
            'output_dir': self.output_dir,
            'dataset_name': self.dataset_name,
            'log2': log2,
            'norm_method': norm_method,
        }
        call_params.update(kwargs)

        result = hlp.scale_data(**call_params)
        if result is None or result.empty:
            log.error(f"Scaling resulted in empty dataset for {self.dataset_name}. Check scaling parameters.")
            sys.exit(1)
        self.scaled_data = result
        log.info(f"Created table: {self._scaled_data_filename}")
        log.info("Created attribute: scaled_data")

    def remove_low_replicable_features(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.remove_low_replicable_features function."""
        if self.check_and_load_attribute('replicate_filtered_data', self._replicate_filtered_data_filename, self.overwrite):
            return

        step = self.normalization_params.get('replicate_handling', {})
        p = step.get('params', step)
        call_params = {
            'data': self.devarianced_data,
            'metadata': self.linked_metadata,
            'dataset_name': self.dataset_name,
            'output_filename': self._replicate_filtered_data_filename,
            'output_dir': self.output_dir,
            'method': step.get('method', 'variance'),
            'group_col': 'group',
            'threshold': p.get('value', 0.5),
            'normalize': True,
            'normalization_scale': 1000000,
            'min_replicates': 2,
        }
        call_params.update(kwargs)

        result = hlp.remove_low_replicable_features(**call_params)
        if result.empty:
            log.error(f"Replicability filtering resulted in empty dataset for {self.dataset_name}. Please adjust replicability parameters.")
            sys.exit(1)
        self.replicate_filtered_data = result
        log.info(f"Created table: {self._replicate_filtered_data_filename}")
        log.info("Created attribute: replicate_filtered_data")

    def plot_pca(self, overwrite: bool = False, analysis_outdir = None, show_plot = True, **kwargs) -> None:
        """Hybrid: Class setup + external hlp.plot_pca function."""
        log.info("Plotting individual PCAs and grid")
        plot_subdir = "pca_plots"
        plot_dir = os.path.join(self.output_dir, plot_subdir)
        os.makedirs(plot_dir, exist_ok=True)

        call_params = {
            'data': {"linked": self.linked_data, "replicate_filtered": self.replicate_filtered_data},
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

# Set up properties for Dataset class
manual_file_storage = {
    'raw_data': 'raw_data.csv',
    'raw_metadata': 'raw_metadata.csv',
    'linked_data': 'linked_data.csv',
    'linked_metadata': 'linked_metadata.csv',
    'filtered_data': 'filtered_data.csv',
    'devarianced_data': 'devarianced_data.csv',
    'scaled_data': 'scaled_data.csv',
    'replicate_filtered_data': 'replicate_filtered_data.csv',
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
        self.chromatography = self.dataset_config['chromatography']
        self.polarity = self.dataset_config['polarity']
        self.mode = "untargeted" # Currently only untargeted supported, not configurable
        self.datatype = "peak-height" # Currently only peak-height supported, not configurable
        if 'metabolite_fraction' in self.dataset_config:
            self.metabolite_fraction = self.dataset_config['metabolite_fraction']
            log.info(f"Using only metabolite fraction '{self.metabolite_fraction}' for MX dataset.")
        else:
            self.metabolite_fraction = None
        self._get_raw_data(overwrite=self.overwrite)
        self._generate_annotation_table(overwrite=self.overwrite)

    def _get_raw_data(self, overwrite: bool = False) -> None:
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
        if result is None or result.empty:
            raise FileNotFoundError(
                f"No MX results data found under {self.dataset_raw_dir}/"
                f"{self.chromatography} for polarity={self.polarity}."
            )

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

    def _get_raw_metadata(self, overwrite: bool = False, superuser: bool = False) -> None:
        log.info("Getting Raw Metadata (MX)")
        if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, self.overwrite):
            log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
            return

        result = hlp.load_mx_metadata(
            input_dir=self.dataset_raw_dir,
            chromatography=self.chromatography,
            output_filename=self._raw_metadata_filename,
            output_dir=self.output_dir,
        )
        self.raw_metadata = result
        log.info(f"\tCreated raw metadata for MX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
        log.info(f"Created table: {self._raw_metadata_filename}")
        log.info("Created attribute: raw_metadata")

    def _generate_annotation_table(self, overwrite: bool = False) -> None:
        """Generate metabolite ID to annotation mapping table for metabolomics data."""
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

class TX(Dataset):
    """Transcriptomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False, last: bool = False, superuser: bool = False):
        super().__init__("tx", project, overwrite, superuser)
        self.index = 1 # Currently only index 1 supported, not configurable
        self.apid = None
        self.genome_type = self.project.config['project']['genome_type']
        self.datatype = "counts" # Currently only counts supported, not configurable
        self._get_raw_data(overwrite=self.overwrite)
        self._generate_annotation_table(overwrite=self.overwrite)

    def _get_raw_data(self, overwrite: bool = False) -> None:
        log.info("Getting Raw Data (TX)")
        counts_path = Path(self.dataset_raw_dir) / f"{self.datatype}.csv"
        self.identifier_column = (
            pd.read_csv(counts_path, nrows=0).columns[0]
            if counts_path.is_file()
            else None
        )
        if self.check_and_load_attribute('raw_data', self._raw_data_filename, self.overwrite):
            if self.identifier_column is None:
                self.identifier_column = self.raw_data.columns[0]
            log.info(f"\t{self.dataset_name} data file with {self.raw_data.shape[0]} samples and {self.raw_data.shape[1]} features.")
            return

        result = hlp.get_tx_data(
            input_dir=self.dataset_raw_dir,
            output_dir=self.output_dir,
            output_filename=self._raw_data_filename,
            type=self.datatype,
            overwrite=overwrite
        )
        if result is None or result.empty:
            raise FileNotFoundError(
                f"No TX {self.datatype} file found under {self.dataset_raw_dir}."
            )
        self.identifier_column = result.columns[0]
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

    def _get_raw_metadata(self, overwrite: bool = False, superuser: bool = False) -> None:
        log.info("Getting Raw Metadata (TX)")
        if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, self.overwrite):
            self.apid = self.raw_metadata['APID'].iloc[0] if 'APID' in self.raw_metadata.columns else None
            log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
            return

        result = hlp.load_raw_metadata(
            input_dir=self.dataset_raw_dir,
            output_dir=self.output_dir,
            output_filename=self._raw_metadata_filename,
        )
        self.apid = result['APID'].iloc[0] if 'APID' in result.columns else None
        self.raw_metadata = result
        log.info(f"\tCreated raw metadata for TX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
        log.info(f"Created table: {self._raw_metadata_filename}")
        log.info("Created attribute: raw_metadata")

    def _generate_annotation_table(self, overwrite: bool = False) -> None:
        """Generate gene annotation table for transcriptomics data."""
        log.info("Generating Gene Annotation Table")
        if self.check_and_load_attribute('annotation_table', self._annotation_table_filename, self.overwrite):
            log.info(f"\tAnnotation table with {self.annotation_table.shape[0]} rows and {self.annotation_table.shape[1]} columns.")
            return

        call_params = {
            'raw_data': self.raw_data,
            'raw_data_dir': self.dataset_raw_dir,
            'genome_type': self.genome_type,
            'output_dir': self.output_dir,
            'output_filename': self._annotation_table_filename,
            'identifier_column': self.identifier_column,
        }

        result = hlp.generate_tx_annotation_table(**call_params)
        if result.empty:
            log.error(f"No annotation table generated for TX dataset. Please check your raw data files.")
            sys.exit(1)
        self.annotation_table = result
        log.info(f"Created table: {self._annotation_table_filename}")
        log.info("Created attribute: annotation_table")

class Analysis(BaseDataHandler):
    """Analysis class with hash-based tagging."""
    
    def __init__(self, project: Project, datasets: list = None, overwrite: bool = False):
        self.project = project
        log.info("Initializing Analysis")
        self.datasets_config = self.project.config['datasets']
        self.analysis_config = self.project.config['analysis']
        self.link_table = self.project.project_config.get('link_table')
        self.overwrite = self.project.overwrite

        # Use hash-based tags for output directory
        self.data_processing_tag = project.data_processing_hash
        self.analysis_tag = project.analysis_hash
        analysis_outdir = self._set_up_analysis_outdir(self.project, self.data_processing_tag, 
                                                      self.analysis_tag, overwrite=self.overwrite)
        self.output_dir = analysis_outdir
        os.makedirs(self.output_dir, exist_ok=True)

        log.info(f"Analysis object created")
        log.info(f"Data processing tag: {self.data_processing_tag}")
        log.info(f"Analysis tag: {self.analysis_tag}")
        log.info(f"Output directory: {self.output_dir}")

        super().__init__(self.output_dir)
        self._setup_analysis_filenames()

        # analysis_config is the flat 'analysis:' block from analysis.yml.
        self.datasets = datasets or []
        if not self.link_table:
            raise ValueError("project.link_table must point to the master link table.")
        linked_metadata = hlp.load_link_table_metadata(
            datasets=self.datasets,
            link_table_path=self.link_table,
        )
        for ds in self.datasets:
            ds.linked_metadata = linked_metadata[ds.dataset_name]

        # Derive integration_mode from the scaling.method configured for each dataset
        # in data_processing.yml.  lfc / moderated_lfc → condition_resolution;
        # everything else → sample_resolution.
        self._integration_mode: str = self._infer_integration_mode(self.datasets, self.datasets_config)

        log.info(f"Track inferred as '{self._integration_mode}' from data_processing.yml scaling.method")
    
        log.info(f"Created analysis with {len(self.datasets)} datasets.")
        for ds in self.datasets:
            log.info(f"\t- {ds.dataset_name} with output directory: {ds.output_dir}")

    # ── LFC scaling methods that imply condition_resolution ──────────────────
    _LFC_METHODS: frozenset = frozenset({"lfc", "moderated_lfc"})

    @staticmethod
    def _infer_integration_mode(datasets: list, datasets_config: dict) -> str:
        """Infer the integration track from each dataset's ``scaling.method``.

        Rules
        -----
        * If **all** datasets use ``lfc`` or ``moderated_lfc`` → ``"condition_resolution"``
        * If **all** datasets use any other method → ``"sample_resolution"``
        * If datasets are **mixed** (some LFC, some not) → ``ValueError``
        * If no datasets are present yet (empty list) → fall back to
          ``"sample_resolution"`` (safe default; will be re-evaluated once
          datasets are attached).

        Parameters
        ----------
        datasets : list
            Dataset objects already attached to the Analysis.
        datasets_config : dict
            The ``datasets`` sub-dict from the project config (used as fallback
            when a dataset object is not yet fully initialised).
        """
        lfc_methods = Analysis._LFC_METHODS

        if not datasets:
            # No datasets attached yet — cannot infer; default to sample_resolution
            log.warning(
                "No datasets attached to Analysis — defaulting to 'sample_resolution'. "
                "Call Analysis again with datasets= to infer the correct track."
            )
            return "sample_resolution"

        modes: list[str] = []
        for ds in datasets:
            # Read scaling.method from the dataset's normalization_params
            scaling_method = (
                ds.normalization_params.get("scaling", {}).get("method", "zscore")
                if hasattr(ds, "normalization_params")
                else datasets_config.get(ds.dataset_name, {})
                    .get("normalization_parameters", {})
                    .get("scaling", {})
                    .get("method", "zscore")
            )
            mode = "condition_resolution" if scaling_method in lfc_methods else "sample_resolution"
            modes.append(mode)

        unique_modes = set(modes)
        if len(unique_modes) > 1:
            details = {
                ds.dataset_name: (
                    ds.normalization_params.get("scaling", {}).get("method", "zscore")
                    if hasattr(ds, "normalization_params") else "unknown"
                )
                for ds in datasets
            }
            raise ValueError(
                "Inconsistent scaling methods across datasets — all datasets must use "
                "either LFC-based methods (lfc, moderated_lfc) for condition_resolution "
                "or non-LFC methods for sample_resolution. "
                f"Found: {details}. "
                "Fix data_processing.yml so all datasets use the same scaling family."
            )

        return unique_modes.pop()

    @property
    def analysis_parameters(self) -> dict:
        """
        Returns the flat analysis config dict directly so that all
        ``self.analysis_parameters.get(...)`` calls work without modification.
        """
        return self.analysis_config

    @property
    def integration_mode(self) -> str:
        """
        The workflow track detected from analysis.yml → analysis.track.

        * ``"sample_resolution"``    — replicates are paired across data types.
          Columns are aligned by Sample ID, row-wise Z-score is applied per
          dataset, matrices are concatenated, and Pearson/Spearman correlation
          followed by HDBSCAN or network clustering identifies co-varying modules.
        * ``"condition_resolution"`` — replicates are not paired.  Replicates are
          collapsed to per-condition means (centroid or LFC sub-approach), then
          row-wise Z-score, concatenation, and clustering are applied.

        All downstream methods use ``self.integrated_data_selected`` as their
        quantitative input regardless of which track is active.
        """
        return self._integration_mode

    @integration_mode.setter
    def integration_mode(self, value: str) -> None:
        valid = {"sample_resolution", "condition_resolution"}
        if value not in valid:
            raise ValueError(f"integration_mode must be one of {valid}, got '{value}'")
        self._integration_mode = value

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
            'feature_network_graph': 'feature_network_graph.graphml',
            'feature_network_edge_table': 'feature_network_edge_table.csv',
            'feature_network_node_table': 'feature_network_node_table.csv',
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

    def filter_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Apply filtering to all datasets in the analysis."""
        log.info("Filtering Data")
        for ds in self.datasets:
            log.info(f"Filtering {ds.dataset_name} dataset...")
            ds.filter_data(overwrite=self.overwrite, **kwargs)

    def devariance_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Apply devariancing to all datasets in the analysis."""
        log.info("Devariancing Data")
        for ds in self.datasets:
            log.info(f"Devariancing {ds.dataset_name} dataset...")
            ds.devariance_data(overwrite=self.overwrite, **kwargs)

    def scale_all_datasets(
        self,
        overwrite: bool = False,
        group_col: str = "group",
        sample_col: str = "unique_group",
        min_reps_for_se: int = 2,
        **kwargs,
    ) -> None:
        """
        Apply per-dataset scaling — the transformation step before concatenation.

        Both tracks are handled here symmetrically:

        * **sample_resolution** — each dataset is log2-transformed and row-wise
          Z-scored independently → ``ds.scaled_data`` (features x samples).

        * **condition_resolution** — each dataset is log2-transformed, replicates
          are collapsed to per-condition medians, and all unique pairwise LFC
          contrasts are computed → ``ds.scaled_data`` (features x contrasts).

        After this step, ``integrate_data()`` simply concatenates the appropriate
        per-dataset attribute across all datasets.
        """
        def _scale_method():
            for ds in self.datasets:
                if ds.check_and_load_attribute(
                    'scaled_data', ds._scaled_data_filename, overwrite or self.overwrite
                ):
                    log.info(
                        f"  [{ds.dataset_name}] scaled_data already exists "
                        f"({ds.scaled_data.shape[0]} features x {ds.scaled_data.shape[1]} contrasts). "
                        "Skipping."
                    )
                    continue
                if not hasattr(ds, 'replicate_filtered_data') or ds.replicate_filtered_data.empty:
                    raise RuntimeError(
                        f"Dataset '{ds.dataset_name}' has no replicate_filtered_data. "
                        "Run replicability_test_all_datasets() first."
                    )
                if not hasattr(ds, 'linked_metadata') or ds.linked_metadata is None or ds.linked_metadata.empty:
                    raise RuntimeError(
                        f"Dataset '{ds.dataset_name}' has no linked_metadata. "
                        "Run link_metadata() first."
                    )

                step = ds.normalization_params.get('scaling', {})
                method = step.get('method', 'zscore')
                params = step.get('params', {})

                if method == 'lfc':
                    log.info(f"  [{ds.dataset_name}] Scaling: log2 → group medians → pairwise LFC...")
                    result = hlp.scale_data_lfc(
                        data=ds.replicate_filtered_data,
                        metadata=ds.linked_metadata,
                        dataset_name=ds.dataset_name,
                        output_filename=ds._scaled_data_filename,
                        output_dir=ds.output_dir,
                        group_col=group_col,
                        sample_col=sample_col,
                    )
                    if result is None or result.empty:
                        raise RuntimeError(
                            f"LFC scaling produced an empty result for '{ds.dataset_name}'."
                        )
                    ds.scaled_data = result
                    log.info(
                        f"  [{ds.dataset_name}] scaled_data: "
                        f"{result.shape[0]} features × {result.shape[1]} contrasts"
                    )

                elif method == 'moderated_lfc':
                    log.info(
                        f"  [{ds.dataset_name}] Scaling: log2 → group medians → "
                        "moderated (shrinkage) pairwise LFC..."
                    )
                    result = hlp.scale_data_moderated_lfc(
                        data=ds.replicate_filtered_data,
                        metadata=ds.linked_metadata,
                        dataset_name=ds.dataset_name,
                        output_filename=ds._scaled_data_filename,
                        output_dir=ds.output_dir,
                        group_col=group_col,
                        sample_col=sample_col,
                        min_reps_for_se=min_reps_for_se
                    )
                    if result is None or result.empty:
                        raise RuntimeError(
                            f"Moderated LFC scaling produced an empty result for '{ds.dataset_name}'."
                        )
                    ds.scaled_data = result
                    log.info(
                        f"  [{ds.dataset_name}] scaled_data: "
                        f"{result.shape[0]} features × {result.shape[1]} contrasts (moderated)"
                    )

                else:
                    # sample_resolution methods: zscore, vst, vsn, rank_normal, modified_zscore, none
                    log.info(f"  [{ds.dataset_name}] Scaling: method='{method}'...")
                    ds.scale_data(
                        overwrite=overwrite,
                        norm_method=method,
                        log2=True,
                        **kwargs
                    )

        _scale_method()
        return

    def replicability_test_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Remove low replicable features from all datasets in the analysis."""
        log.info("Removing Unreplicable Features")
        for ds in self.datasets:
            log.info(f"Filtering outlier replicates from {ds.dataset_name} dataset (output: replicate_filtered_data)...")
            ds.remove_low_replicable_features(overwrite=self.overwrite, **kwargs)

    def plot_pca_all_datasets(self, overwrite: bool = False, show_plot: bool = True, **kwargs) -> None:
        """Plot PCA for all datasets in the analysis."""
        log.info("Plotting Individual PCAs and Grid")
        for ds in self.datasets:
            log.info(f"Plotting PCA for {ds.dataset_name} dataset...")
            ds.plot_pca(overwrite=self.overwrite,
                        analysis_outdir=self.output_dir,
                        show_plot=show_plot,
                        **kwargs)

    def link_data(self, overlap_only: bool = True, overwrite: bool = False) -> None:
        """Rename raw quantitative sample columns using the master link table."""
        log.info("Linking analysis datasets along shared samples")

        # Check if all datasets already have linked data
        datasets_to_process = [
            ds for ds in self.datasets
            if not ds.check_and_load_attribute('linked_data', ds._linked_data_filename, self.overwrite)
        ]
        if not datasets_to_process:
            return

        # The master link table is loaded during Analysis initialization.
        for ds in self.datasets:
            if not hasattr(ds, 'linked_metadata') or ds.linked_metadata.empty:
                raise ValueError(f"Dataset {ds.dataset_name} lacks link-table metadata.")

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

    def plot_dataset_distributions(
        self,
        bins: int = 50,
        transparency: float = 0.5,
        xlog: bool = False,
        ylog: bool = False,
    ) -> None:
        """Plot histograms of feature values for each dataset in the analysis."""
        log.info("Plotting feature value distributions for all datasets")
        dataframes = {ds.dataset_name: ds.scaled_data for ds in self.datasets if hasattr(ds, "scaled_data")}

        hlp.plot_data_variance_histogram(
            dataframes=dataframes,
            output_dir=self.output_dir,
            bins=bins,
            transparency=transparency,
            xlog=xlog,
            ylog=ylog
        )

    def plot_integrated_pca(self) -> None:
        """Plot histograms of feature values for each dataset in the analysis."""
        log.info("Plotting PCA of integrated features")

        hlp.plot_simple_pca(
            df=self.integrated_data_selected,
            metadata=self.integrated_metadata,
            title="Integrated Data PCA",
            output_dir=self.output_dir,
        )

    def integrate_metadata(
        self,
        group_col: str = "group",
        overlap_only: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Hybrid: Class validation + external hlp.integrate_metadata function."""
        track = self.integration_mode
        log.info(f"Integrating metadata across data types (track='{track}')")

        # sample_resolution produces sample-level metadata; reuse cache safely
        can_reuse_cached = (
            track == "sample_resolution"
            and self.check_and_load_attribute(
                "integrated_metadata",
                self._integrated_metadata_filename,
                overwrite or self.overwrite
            )
        )
        if can_reuse_cached:
            log.info(
                f"\tIntegrated metadata object 'integrated_metadata' with "
                f"{self.integrated_metadata.shape[0]} rows and {self.integrated_metadata.shape[1]} columns."
            )
            return

        # Map new track names to the helper's method parameter
        helper_method = 'replicate_matched' if track == 'sample_resolution' else 'lfc'

        result = hlp.integrate_metadata(
            datasets=self.datasets,
            metadata_vars=self.project.study_variables,
            unifying_col="unique_group",
            output_filename=self._integrated_metadata_filename,
            output_dir=self.output_dir,
            method=helper_method,
            group_col=group_col,
            overlap_only=overlap_only,
        )

        if result.empty:
            log.error("Integrating metadata resulted in empty table. Please check datasets and parameters.")
            sys.exit(1)

        self.integrated_metadata = result
        log.info(
            f"Created integrated metadata table with {self.integrated_metadata.shape[0]} rows "
            f"and {self.integrated_metadata.shape[1]} columns."
        )
        log.info(f"Created table: {self._integrated_metadata_filename}")
        log.info("Created attribute: integrated_metadata")

    def integrate_data(
        self,
        overlap_only: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Concatenate per-dataset scaled matrices into a single integrated matrix.

        Both tracks are handled symmetrically — ``scale_all_datasets()`` performs
        all transformation logic; this method only concatenates:

        * **sample_resolution** — concatenates ``ds.scaled_data``
          (features x samples) across datasets.
        * **condition_resolution** — concatenates ``ds.scaled_data``
          (features x contrasts) across datasets.
        """
        track = self.integration_mode

        if self.check_and_load_attribute(
            "integrated_data",
            self._integrated_data_filename,
            overwrite or self.overwrite
        ):
            log.info(
                f"\tIntegrated data object 'integrated_data' with "
                f"{self.integrated_data.shape[0]} features and {self.integrated_data.shape[1]} columns."
            )
            self.integration_mode = track
            return

        if track == 'sample_resolution':
            data_attr = 'scaled_data'
            log.info("Integrating data matrices (sample_resolution: concatenating scaled_data)...")
            for ds in self.datasets:
                if not hasattr(ds, 'scaled_data') or ds.scaled_data is None or ds.scaled_data.empty:
                    raise RuntimeError(
                        f"Dataset '{ds.dataset_name}' has no scaled_data. "
                        "Run analysis.scale_all_datasets() before integrate_data()."
                    )
        else:
            data_attr = 'scaled_data'
            log.info("Integrating data matrices (condition_resolution: concatenating scaled_data)...")
            for ds in self.datasets:
                if not hasattr(ds, 'scaled_data') or ds.scaled_data is None or ds.scaled_data.empty:
                    raise RuntimeError(
                        f"Dataset '{ds.dataset_name}' has no scaled_data. "
                        "Run analysis.scale_all_datasets() before integrate_data()."
                    )

        result = hlp.integrate_data(
            datasets=self.datasets,
            overlap_only=overlap_only,
            output_filename=self._integrated_data_filename,
            output_dir=self.output_dir,
            data_attr=data_attr,
        )

        if result.empty:
            log.error("Integrating data resulted in empty table. Please check datasets and parameters.")
            sys.exit(1)

        self.integrated_data = result
        self.integration_mode = track
        log.info(
            f"Created integrated data table with {self.integrated_data.shape[0]} features "
            f"and {self.integrated_data.shape[1]} columns."
        )
        log.info(f"track set to '{self.integration_mode}'")
        log.info(f"Created table: {self._integrated_data_filename}")
        log.info("Created attribute: integrated_data")

    def annotate_integrated_features(self, overlap_only: bool = True, overwrite: bool = False) -> pd.DataFrame:
        """Hybrid: Class orchestration + external annotate_integrated_features function."""
        log.info("Annotating integrated features")
        if self.check_and_load_attribute('feature_annotation_table', self._feature_annotation_table_filename, self.overwrite):
            log.info(f"\tAnnotated features object 'feature_annotation_table' with {self.feature_annotation_table.shape[0]} features and {self.feature_annotation_table.shape[1]} samples.")
            return

        annotation_df = hlp.annotate_integrated_features(
            integrated_data=self.integrated_data,
            datasets=self.datasets,
            output_dir=self.output_dir,
            cache_dir=self.project.cache_dir,
            output_filename=self._feature_annotation_table_filename,
        )

        if annotation_df.empty:
            log.error(f"Annotating integrated features resulted in empty table. Please check your datasets and annotation maps.")
            sys.exit(1)
        self.feature_annotation_table = annotation_df
        log.info(f"Created an annotated integrated features table with {self.feature_annotation_table.shape[0]} entries ({len(self.feature_annotation_table['feature_id'].unique())} unique features) and {self.feature_annotation_table.shape[1]} samples.")
        log.info(f"Created table: {self._feature_annotation_table_filename}")
        log.info("Created attribute: feature_annotation_table")

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
            data=self.integrated_data_selected,
            metadata=self.integrated_metadata,
            node_table=self.feature_network_node_table,
            submodule_name=submodule_name,
            metadata_group=metadata_cat,
            output_dir=self.output_dir,
            save_plot=save_plot
        )

    def perform_feature_selection(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.perform_feature_selection function."""
        if self.check_and_load_attribute('integrated_data_selected', self._integrated_data_selected_filename, self.overwrite):
            log.info(f"\tFeature selection data object 'integrated_data_selected' with {self.integrated_data_selected.shape[0]} features and {self.integrated_data_selected.shape[1]} samples.")
            return

        feature_selection_params = self.analysis_parameters.get('feature_selection', {})
        log.info(f"Subsetting Features using the {feature_selection_params.get('method', 'unspecified')} method with parameters: {feature_selection_params.get('params', {})}")
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

    def run_full_network_analyzer(self, overwrite: bool = False, **kwargs) -> None:
        grouping_params = self.analysis_parameters.get('feature_grouping', {}).get('params', {})
        output_dir = os.path.join(self.output_dir, "network_analyzer_results")
        results = hlp.compare_network_topologies(
            integrated_data=self.integrated_data_selected,
            feature_prefixes=[ds.dataset_name + "_" for ds in self.datasets],
            correlation_params=grouping_params,
            network_params=grouping_params,
            annotation_input=self.feature_annotation_table,
            output_dir=output_dir,
            overwrite=overwrite,
            plot_interactive=grouping_params.get('show_network_plot', False),
        )

        return results

    def calculate_correlated_features(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.calculate_correlated_features function."""
        log.info("Calculating Correlated Features")

        feature_grouping_params = self.analysis_parameters.get('feature_grouping', {})
        selected_method = feature_grouping_params.get('method', 'network_modules')
        correlation_params = feature_grouping_params.get('params', {})

        if selected_method != 'network_modules':
            log.info(f"Selected feature grouping method is '{selected_method}'; skipping correlation computation (not required).")
            return

        if self.check_and_load_attribute('feature_correlation_table', self._feature_correlation_table_filename, self.overwrite):
            log.info(f"\tFeature correlation table object 'feature_correlation_table' with {self.feature_correlation_table.shape[0]} feature pairs.")
            return

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
            'corr_mode': correlation_params.get('corr_mode', 'bipartite'),
            'calculate_r2': True
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

    def group_features_step(self, overwrite: bool = False, **kwargs) -> None:
        """Unified feature grouping dispatcher supporting network_modules, hierarchical_clustering, hdbscan, nmf, leiden_knn, wgcna."""
        log.info("Grouping Features")
        submodule_subdir = "submodules"
        submodule_dir = os.path.join(self.output_dir, submodule_subdir)
        os.makedirs(submodule_dir, exist_ok=True)

        feature_grouping_params = self.analysis_parameters.get('feature_grouping', {})
        selected_method = feature_grouping_params.get('method', 'network_modules')
        method_params = feature_grouping_params.get('params', {})

        # Check if outputs already exist
        if self.check_and_load_attribute('feature_network_node_table', self._feature_network_node_table_filename, self.overwrite) and \
            self.check_and_load_attribute('feature_network_edge_table', self._feature_network_edge_table_filename, self.overwrite):
            if selected_method == 'network_modules':
                log.info("Displaying existing network visualization...")
                hlp.display_existing_network(
                    graph_file=self._feature_network_graph_filename,
                    node_table=self.feature_network_node_table,
                    edge_table=self.feature_network_edge_table,
                    network_layout=method_params.get('network_layout', None)
                )
            else:
                log.info(f"Feature grouping table already exists ({selected_method}). Loaded from disk.")
            return
        else:
            hlp.clear_directory(submodule_dir)

        output_filenames = {
            'graph': self._feature_network_graph_filename,
            'node_table': self._feature_network_node_table_filename,
            'edge_table': self._feature_network_edge_table_filename,
            'submodule_path': submodule_dir
        }

        # For network_modules, feature_correlation_table must exist
        feature_correlation_table = None
        if selected_method == 'network_modules':
            if not hasattr(self, 'feature_correlation_table') or self.feature_correlation_table is None:
                log.error("feature_correlation_table is required for network_modules grouping. Run calculate_correlated_features() first.")
                sys.exit(1)
            feature_correlation_table = self.feature_correlation_table

        call_params = {
            'data': self.integrated_data_selected,
            'method': selected_method,
            'method_params': method_params,
            'output_dir': self.output_dir,
            'output_filenames': output_filenames,
            'datasets': self.datasets,
            'annotation_df': self.feature_annotation_table,
            'integrated_data': self.integrated_data_selected,
            'integrated_metadata': self.integrated_metadata,
            'feature_correlation_table': feature_correlation_table,
        }
        call_params.update(kwargs)

        node_table, edge_table = hlp.group_features(**call_params)

        if node_table.empty:
            log.error(f"Feature grouping resulted in empty node table. Please check your data and grouping parameters.")
            sys.exit(1)

        self.feature_network_node_table = node_table
        self.feature_network_edge_table = edge_table

        log.info(f"Created table: {self._feature_network_node_table_filename}")
        log.info("Created attribute: feature_network_node_table")
        log.info(f"Created table: {self._feature_network_edge_table_filename}")
        log.info("Created attribute: feature_network_edge_table")

    def group_features(self, overwrite: bool = False, **kwargs) -> None:
        """
        Public entry point for feature grouping.

        Reads ``feature_grouping.method`` from the analysis config and dispatches
        to the appropriate grouping backend:

        * ``network_modules``         - pairwise Pearson/Spearman correlation → graph →
          Louvain/Leiden community detection.  Requires
          ``calculate_correlated_features()`` to have been run first (called
          automatically if not yet present).
        * ``hierarchical_clustering`` - scipy agglomerative clustering on feature vectors.
        * ``hdbscan``                 - HDBSCAN density-based clustering.
        * ``nmf``                     - Non-negative Matrix Factorization with automatic
          component count selection via reconstruction-error elbow.
        * ``leiden_knn``              - Leiden community detection on an approximate
          k-nearest-neighbor graph (pynndescent); resolution auto-selected by
          maximising graph modularity.
        * ``wgcna``                   - Weighted Gene Co-expression Network Analysis.
          Builds a soft-thresholded adjacency matrix, computes the Topological
          Overlap Matrix (TOM), clusters features, and detects modules.  Writes
          full WGCNA diagnostic outputs (soft-threshold plot, dendrogram,
          module eigengenes, kME, kIN, module-trait heatmap, GS vs MM plots)
          to ``<output_dir>/wgcna_results/``.

        All methods write a node table (``feature_network_node_table``) with a
        unified ``group`` column and an edge table (``feature_network_edge_table``)
        to disk, and store them as attributes on the Analysis object.

        For ``network_modules`` the correlation step is run automatically if
        ``feature_correlation_table`` is not yet present.  For all other methods
        (including ``wgcna``) the correlation step is skipped.
        """
        feature_grouping_params = self.analysis_parameters.get('feature_grouping', {})
        selected_method = feature_grouping_params.get('method', 'network_modules')

        # For network_modules, run the correlation step first if needed
        if selected_method == 'network_modules':
            if not hasattr(self, 'feature_correlation_table') or \
               (hasattr(self, 'feature_correlation_table') and self.feature_correlation_table.empty):
                log.info("network_modules selected — running calculate_correlated_features() first...")
                self.calculate_correlated_features(overwrite=overwrite)

        self.group_features_step(overwrite=overwrite, **kwargs)

    def compare_groups_to_pathways(self, **override_kwargs) -> dict:
        """
        Compare data-driven feature groups against knowledge-driven pathway annotations.

        All parameters are read from the ``pathway_enrichment`` block in
        ``analysis.yml``.  Any keyword arguments passed directly to this method
        override the config values.

        Pipeline: (1) explode and filter the feature/pathway universe, (2) 
        compute hypergeometric enrichment matrices, (3) select top pathways 
        for display, (4) calculate aggregated abundance trends for rows and 
        columns, (5) plot results as a series of heatmaps with attached 
        trend tracks.

        Config keys (under ``analysis.pathway_enrichment``):
            ``pathway_col``             - annotation column to use (default "modelseed_pathway")
            ``bipartite_only``          - keep only pathways with features from every data type
            ``min_features_per_pathway``- drop pathways smaller than this before testing (default 3)
            ``min_features_per_group``  - drop groups smaller than this before testing (default 5)
            ``alpha``                   - significance threshold for enrichment (default 0.05)
            ``top_n``                   - number of pathways to display in the heatmap (default 50)
            ``trend_sort_by``           - metadata column to order trend track columns by
            ``trend_collapse_by``       - metadata column to pool samples by (e.g. "timepoint")
            ``trend_agg``               - "mean" | "median" summary shown as trend track color
            ``trend_cmap``              - colormap for trend tracks (default "RdBu_r")

        Returns a dict with keys:
            ``figs`` (the heatmap figures), ``pval_matrix`` (full stats table)
        """
        if not hasattr(self, 'feature_network_node_table') or self.feature_network_node_table.empty:
            raise RuntimeError(
                "feature_network_node_table is empty. Run analysis.group_features() first."
            )

        # Retrieve config block
        pe = self.analysis_parameters.get('pathway_enrichment', {})

        # We pass the dataframes and the config dict directly to the orchestrator.
        # The orchestrator will merge pe with override_kwargs.
        call_params = {
            'node_table':        self.feature_network_node_table,
            'quant_df':          self.integrated_data_selected,
            'metadata_df':       self.integrated_metadata,
            'output_dir':        self.output_dir,
            'pe':                pe,
        }
        
        # Allow user to override any pe parameter or orchestration setting
        call_params.update(override_kwargs)

        return hlp.compare_groups_to_pathways(**call_params)

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
}
#for attr in config['analysis']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Analysis, attr, Analysis._create_property(None, attr, f'_{attr}_filename'))