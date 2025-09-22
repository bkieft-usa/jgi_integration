import pandas as pd

def link_metadata_tables(dataset_info):

    linked_metadata = {}
    for dataset_name, dataset_dict in dataset_info.items():
        if 'tx' in dataset_name:
            print(f"Linking metadata for dataset: {dataset_name}")
            print(f"Reading metadata from: {dataset_dict['outdir']}/{dataset_dict['raw']}")
            full_metadata = pd.read_csv(f"{dataset_dict['outdir']}/{dataset_dict['raw']}")
            full_metadata = full_metadata[full_metadata['APID'].astype(str) == str(dataset_dict['apid'])]
            full_metadata['group'] = full_metadata['sample_name'].str.split('-').str[:2].str.join('-')
            full_metadata['group'] = full_metadata['group'].str.replace('^10.', 'LowS-', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('^700.', 'HighS-', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('-10-', '-LowL-', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('-100-', '-HighL-', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('h$', 'h-HeatStr', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('-R\\+24h-HeatStr', '-24h-Recov', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('-', '_', regex=True)
            full_metadata['replicate'] = full_metadata['sample_name'].str.split('-').str[2]
            full_metadata['salinity'] = full_metadata['group'].str.split('_').str[0]
            full_metadata['light'] = full_metadata['group'].str.split('_').str[1]
            full_metadata['timepoint'] = full_metadata['group'].str.split('_').str[2]
            full_metadata['condition'] = full_metadata['group'].str.split('_').str[3]
            full_metadata.loc[full_metadata['timepoint'] == '0h', 'condition'] = 'Initial'
            full_metadata['group'] = full_metadata['salinity'] + '_' + \
                                                full_metadata['light'] + '_' + \
                                                full_metadata['timepoint'] + '_' + \
                                                full_metadata['condition']
            full_metadata['unique_group'] = full_metadata['group'] + '_' + \
                                                full_metadata['replicate']
            full_metadata = full_metadata[full_metadata['timepoint'] != '1h']
            print(f"    Created linked metadata with {full_metadata.shape[0]} samples and {full_metadata.shape[1]} metadata fields.")
            linked_metadata[dataset_name] = full_metadata

        elif 'mx' in dataset_name:
            print(f"Linking metadata for dataset: {dataset_name}")
            print(f"Reading metadata from: {dataset_dict['outdir']}/{dataset_dict['raw']}")
            full_metadata = pd.read_csv(f"{dataset_dict['outdir']}/{dataset_dict['raw']}")
            full_metadata['group'] = full_metadata['file'].str.split('_').str[12]
            full_metadata['group'] = full_metadata['group'].str.replace('^M-', '', regex=True)
            full_metadata['group'] = full_metadata['group'].str.replace('-', '_', regex=True)
            full_metadata['chromatography'] = full_metadata['file'].str.split('_').str[7]
            full_metadata['chromatography'] = full_metadata['chromatography'].apply(lambda x: 'polar' if 'HILIC' in x else 'nonpolar')
            full_metadata['replicate'] = full_metadata['file'].str.split('_').str[13]
            full_metadata['salinity'] = full_metadata['group'].str.split('_').str[0]
            full_metadata['light'] = full_metadata['group'].str.split('_').str[1]
            full_metadata['timepoint'] = full_metadata['group'].str.split('_').str[2]
            full_metadata['condition'] = full_metadata['group'].str.split('_').str[3]
            full_metadata.loc[full_metadata['timepoint'] == '0h', 'condition'] = 'Initial'
            full_metadata['group'] = full_metadata['salinity'] + '_' + \
                                                full_metadata['light'] + '_' + \
                                                full_metadata['timepoint'] + '_' + \
                                                full_metadata['condition']
            full_metadata['unique_group'] = full_metadata['group'] + '_' + \
                                                full_metadata['replicate']
            full_metadata = full_metadata[~full_metadata['file'].str.contains('media|ExCtrl|QC')]
            print(f"    Created linked metadata with {full_metadata.shape[0]} samples and {full_metadata.shape[1]} metadata fields.")
            linked_metadata[dataset_name] = full_metadata

        else:
            raise ValueError(f"Dataset name '{dataset_name}' does not match expected patterns for input datasets.")
        
    return linked_metadata