import os
import sys
import yaml
import glob
import pandas as pd
import numpy as np
import shutil
from typing import Dict, Any, List, Optional
from IPython.display import display
import tools.helpers as hlp
import logging

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "\033[47m%(levelname)s – %(message)s\033[0m"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

class Project:
    """Project configuration and directory management."""

    def __init__(self, config):
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
        log.info("Initializing Datasets")
        self.project = project
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
        #self._setup_dataset_filenames(self.datasets_config['file_storage'])
        self._setup_dataset_filenames()

        # Configuration
        self.normalization_params = self.dataset_config.get('normalization_parameters', {})
        self.annotation = self.dataset_config.get('annotation', {})

    @staticmethod
    def set_up_dataset_outdir(project: Project, datasets_config: dict, dataset_config: dict, dataset_name: str, overwrite: bool = False) -> bool:
        """Check if the Dataset_Processing--[tag] directory already exists."""
        processing_dir = os.path.join(
            project.project_dir,
            f"Dataset_Processing--{datasets_config['data_processing_tag']}",
            dataset_config['dataset_dir']
        )
        if os.path.exists(processing_dir):
            if overwrite:
                log.info(f"Overwriting existing dataset processing directory for {dataset_name} at {processing_dir}.")
                #shutil.rmtree(processing_dir)
                return processing_dir
            else:
                log.info(f"ERROR: Dataset processing directory already exists for {dataset_name} at {processing_dir}")
                log.info("\nPlease choose a different tag, delete the existing directory, or use the overwrite=True flag before proceeding.")
                sys.exit(1)
        else:
            log.info(f"Set up {dataset_name} dataset output directory: {processing_dir}")
            return processing_dir

    #def _setup_dataset_filenames(self, file_storage):
    def _setup_dataset_filenames(self):
        """Setup all dataset filename attributes."""
        #for attr, filename in file_storage.items():
        #    setattr(self, f"_{attr}_filename", filename)
        manual_file_storage = {
            'raw_data': 'raw_data.csv',
            'raw_metadata': 'raw_metadata.csv',
            'linked_data': 'linked_data.csv',
            'linked_metadata': 'linked_metadata.csv',
            'filtered_data': 'filtered_data.csv',
            'devarianced_data': 'devarianced_data.csv',
            'scaled_data': 'scaled_data.csv',
            'normalized_data': 'normalized_data.csv',
            'pca_grid': 'pca_grid.pdf'
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
        self.filtered_data = result
    
    def devariance_data(self, overwrite: bool = False, **kwargs) -> None:
        """Remove low-variance features using external helper function with class integration."""
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
        self.devarianced_data = result

    def scale_data(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.scale_data function."""
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
        self.scaled_data = result
    
    def remove_low_replicable_features(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.remove_low_replicable_features function."""
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
        self.normalized_data = result
    
    def plot_pca(self, overwrite: bool = False, analysis_outdir = None, show_plot = True, **kwargs) -> None:
        """Hybrid: Class setup + external hlp.plot_pca function."""
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
    'pca_grid': 'pca_grid.pdf'
}
#for attr in config['datasets']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Dataset, attr, Dataset._create_property(None, attr, f'_{attr}_filename'))

class MX(Dataset):
    """Metabolomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False):
        super().__init__("mx", project, overwrite)
        self.chromatography = self.dataset_config['chromatography']
        self.polarity = self.dataset_config['polarity']
        #self.mode = self.dataset_config['mode']
        self.mode = "untargeted" # Currently only untargeted supported, not configurable
        #self.datatype = self.dataset_config['datatype']
        self.datatype = "peak-height" # Currently only peak-height supported, not configurable

    def get_raw_data(self, overwrite: bool = False) -> None:
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
        self.raw_data = result
        log.info(f"\tCreated raw data for MX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.\n")

    def get_raw_metadata(self, overwrite: bool = False) -> None:
        log.info("Getting Raw Metadata (MX)")
        if self.check_and_load_attribute('raw_metadata', self._raw_metadata_filename, overwrite):
            log.info(f"\t{self.dataset_name} metadata file with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.")
            return

        mx_data_pattern = f"{self.dataset_raw_dir}/*{self.chromatography}*/*_{self.datatype}.csv"
        if not glob.glob(mx_data_pattern) or overwrite:
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
        self.raw_metadata = result
        log.info(f"\tCreated raw metadata for MX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.\n")


class TX(Dataset):
    """Transcriptomics dataset with specific configuration."""
    def __init__(self, project: Project, overwrite: bool = False):
        super().__init__("tx", project, overwrite)
        #self.index = self.dataset_config['index']
        self.index = 1 # Currently only index 1 supported, not configurable
        self.apid = None
        #self.datatype = self.dataset_config['datatype']
        self.datatype = "counts" # Currently only counts supported, not configurable

    def get_raw_data(self, overwrite: bool = False) -> None:
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
        self.raw_data = result
        log.info(f"\tCreated raw data for TX with {self.raw_data.shape[1]} samples and {self.raw_data.shape[0]} features.\n")

    def get_raw_metadata(self, overwrite: bool = False) -> None:
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
        self.raw_metadata = result
        log.info(f"\tCreated raw metadata for TX with {self.raw_metadata.shape[0]} samples and {self.raw_metadata.shape[1]} metadata fields.\n")

class Analysis(BaseDataHandler):
    """Simplified Analysis class using consistent hybrid approach."""

    def __init__(self, project: Project, datasets: list = None, overwrite: bool = False):
        log.info("Initializing Analysis")
        self.project = project
        self.datasets_config = self.project.config['datasets']
        self.analysis_config = self.project.config['analysis']
        self.metadata_link_script = self.project.project_config['metadata_link']

        # Check if analysis directory exists
        analysis_outdir = self.set_up_analysis_outdir(self.project, self.datasets_config, 
                                                      self.analysis_config, overwrite=overwrite)
        self.output_dir = analysis_outdir
        os.makedirs(self.output_dir, exist_ok=True)
        super().__init__(self.output_dir)

        #self._setup_analysis_filenames(self.analysis_config['file_storage'])
        self._setup_analysis_filenames()

        self.analysis_parameters = self.analysis_config.get('analysis_parameters', {})
        self.datasets = datasets or []
    
        log.info(f"Created analysis with {len(self.datasets)} datasets.")
        for ds in self.datasets:
            log.info(f"\t- {ds.dataset_name} with output directory: {ds.output_dir}")

    @staticmethod
    def set_up_analysis_outdir(project: Project, datasets_config: dict, analysis_config: dict, overwrite: bool = False) -> str:
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
                log.info(f"ERROR: Analysis directory already exists for {dataset_name} at {analysis_dir}")
                log.info("\nPlease choose a different tag, delete the existing directory, or use the overwrite=True flag before proceeding.")
                sys.exit(1)
        else:
            log.info(f"Set up analysis output directory: {analysis_dir}")
            return analysis_dir

    #def _setup_analysis_filenames(self, file_storage):
    def _setup_analysis_filenames(self):
        """Setup analysis filename attributes."""
        #for attr, filename in file_storage.items():
        #    setattr(self, f"_{attr}_filename", filename)
        manual_file_storage = {
            'integrated_metadata': 'integrated_metadata.csv',
            'integrated_data': 'integrated_data.csv',
            'feature_annotation_table': 'feature_annotation_table.csv',
            'integrated_data_selected': 'integrated_data_selected.csv',
            'feature_correlation_table': 'feature_correlation_table.csv',
            'feature_network_graph': 'feature_network_graph.graphml',
            'feature_network_edge_table': 'feature_network_edge_table.csv',
            'feature_network_node_table': 'feature_network_node_table.csv',
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

    def filter_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Apply filtering to all datasets in the analysis."""
        log.info("Filtering Data")
        for ds in self.datasets:
            log.info(f"Filtering {ds.dataset_name} dataset...")
            ds.filter_data(overwrite=overwrite, **kwargs)

    def devariance_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Apply devariancing to all datasets in the analysis."""
        log.info("Devariancing Data")
        for ds in self.datasets:
            log.info(f"Devariancing {ds.dataset_name} dataset...")
            ds.devariance_data(overwrite=overwrite, **kwargs)

    def scale_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Apply scaling to all datasets in the analysis."""
        log.info("Scaling Data")
        for ds in self.datasets:
            log.info(f"Scaling {ds.dataset_name} dataset...")
            ds.scale_data(overwrite=overwrite, **kwargs)

    def replicability_test_all_datasets(self, overwrite: bool = False, **kwargs) -> None:
        """Remove low replicable features from all datasets in the analysis."""
        log.info("Removing Unreplicable Features")
        for ds in self.datasets:
            log.info(f"Removing low replicable features from {ds.dataset_name} dataset...")
            ds.remove_low_replicable_features(overwrite=overwrite, **kwargs)

    def plot_pca_all_datasets(self, overwrite: bool = False, show_plot: bool = True, **kwargs) -> None:
        """Plot PCA for all datasets in the analysis."""
        log.info("Plotting Individual PCAs and Grid")
        for ds in self.datasets:
            log.info(f"Plotting PCA for {ds.dataset_name} dataset...")
            ds.plot_pca(overwrite=overwrite, 
                        analysis_outdir=self.output_dir, 
                        show_plot=show_plot, 
                        **kwargs)

    def link_metadata(self, overwrite: bool = False) -> None:
        """Hybrid: Class orchestration + external hlp.link_metadata_with_custom_script function."""

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
            ds.linked_metadata = linked_metadata[ds.dataset_name]
            log.info(f"Created linked_metadata for {ds.dataset_name} with {ds.linked_metadata.shape[0]} samples and {ds.linked_metadata.shape[1]} metadata fields.\n")

    def link_data(self, overlap_only: bool = True, overwrite: bool = False) -> None:
        """Hybrid: Class orchestration + external hlp.link_data_across_datasets function."""

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
            ds.linked_data = linked_data[ds.dataset_name]
            log.info(f"Created linked_data for {ds.dataset_name} with {ds.linked_data.shape[1]} samples and {ds.linked_data.shape[0]} features.\n")
        
        return

    def plot_dataset_distributions(self, datatype: str = "normalized", bins: int = 50, transparency: float = 0.8, xlog: bool = False, show_plot: bool = True) -> None:
        """
        Plot histograms of feature values for each dataset in the analysis.
        """
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

    def integrate_metadata(self, overwrite: bool = False) -> None:
        """Hybrid: Class validation + external hlp.integrate_metadata function."""
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
        self.integrated_metadata = result
        log.info(f"Created a single integrated metadata table with {self.integrated_metadata.shape[0]} samples and {self.integrated_metadata.shape[1]} metadata fields.\n")
    
    def integrate_data(self, overlap_only: bool = True, overwrite: bool = False) -> None:
        """Hybrid: Class validation + external hlp.integrate_data function."""
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
        self.integrated_data = result
        log.info(f"Created a single integrated data table with {self.integrated_data.shape[0]} samples and {self.integrated_data.shape[1]} features.\n")

    def perform_feature_selection(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.perform_feature_selection function."""
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
        self.integrated_data_selected = result
        log.info(f"Created a subset of the integrated data with {self.integrated_data_selected.shape[0]} samples and {self.integrated_data_selected.shape[1]} features for network analysis.\n")

    def calculate_correlated_features(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class validation + external hlp.calculate_correlated_features function."""
        log.info("Calculating Correlated Features")
        if self.check_and_load_attribute('feature_correlation_table', self._feature_correlation_table_filename, overwrite):
            log.info(f"\tFeature correlation table object 'feature_correlation_table' with {self.feature_correlation_table.shape[0]} feature pairs.")
            return
        
        # Get parameters from config with defaults
        correlation_params = self.analysis_parameters.get('correlation', {})
        networking_params = self.analysis_parameters.get('networking', {})        
        call_params = {
            'data': self.integrated_data_selected,
            'output_filename': self._feature_correlation_table_filename,
            'output_dir': self.output_dir,
            'method': correlation_params.get('corr_method', 'pearson'),
            'cutoff': correlation_params.get('corr_cutoff', 0.5),
            'keep_negative': correlation_params.get('keep_negative', False),
            'block_size': correlation_params.get('block_size', 500),
            'n_jobs': correlation_params.get('cores', -1),
            'only_bipartite': networking_params.get('network_mode', 'bipartite') == 'bipartite'
        }
        call_params.update(kwargs)
        
        result = hlp.bipartite_correlation(**call_params)
        self.feature_correlation_table = result
        log.info(f"Created a feature correlation table with {self.feature_correlation_table.shape[0]} feature pairs.\n")

    def plot_correlation_network(self, overwrite: bool = False, **kwargs) -> None:
        log.info("Plotting Correlation Network")
        network_subdir = "feature_network"
        submodule_subdir = "submodules"
        network_dir = os.path.join(self.output_dir, network_subdir)
        submodule_dir = os.path.join(network_dir, submodule_subdir)
        os.makedirs(network_dir, exist_ok=True)
        os.makedirs(submodule_dir, exist_ok=True)
        if self.check_and_load_attribute('feature_network_graph', os.path.join(network_subdir, self._feature_network_graph_filename), overwrite) or \
            self.check_and_load_attribute('feature_network_node_table', os.path.join(network_subdir, self._feature_network_node_table_filename), overwrite) or \
                self.check_and_load_attribute('feature_network_edge_table', os.path.join(network_subdir, self._feature_network_edge_table_filename), overwrite):
            return
        else: # necessary to clear carryover from previous analyses
            hlp.clear_directory(network_dir)
            hlp.clear_directory(submodule_dir)

        networking_params = self.analysis_parameters.get('networking', {})
        correlation_params = self.analysis_parameters.get('correlation', {})
        output_filenames = {
            'graph': os.path.join(network_dir, self._feature_network_graph_filename),
            'node_table': os.path.join(network_dir, self._feature_network_node_table_filename),
            'edge_table': os.path.join(network_dir, self._feature_network_edge_table_filename),
            'submodule_path': submodule_dir
        }

        call_params = {
            'corr_table': self.feature_correlation_table,
            'feature_prefixes': [ds.dataset_name for ds in self.datasets],
            'integrated_data': self.integrated_data_selected,
            'integrated_metadata': self.integrated_metadata,
            'output_filenames': output_filenames,
            'annotation_df': getattr(self, 'feature_annotation_table', None),
            'network_mode': networking_params.get('network_mode', 'bipartite'),
            'submodule_mode': networking_params.get('submodule_mode', 'community'),
            'show_plot': networking_params.get('interactive_plot', False),
            'interactive_layout': networking_params.get('interactive_layout', None),
            'corr_cutoff': correlation_params.get('corr_cutoff', 0.5),
            'wgcna_params': networking_params.get('wgcna_params', {"beta": 5, "min_module_size": 10, "distance_cutoff": 0.25})
        }
        call_params.update(kwargs)
        
        hlp.plot_correlation_network(**call_params)
        log.info("Created correlation network graph and associated node/edge tables.\n")

    def run_mofa2_analysis(self, overwrite: bool = False, **kwargs) -> None:
        """Hybrid: Class parameter setup + external hlp.run_full_mofa2_analysis function."""
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
    'mofa_model': 'mofa_model.hdf5'
}
#for attr in config['analysis']['file_storage']:
for attr, filename in manual_file_storage.items():
    setattr(Analysis, attr, Analysis._create_property(None, attr, f'_{attr}_filename'))