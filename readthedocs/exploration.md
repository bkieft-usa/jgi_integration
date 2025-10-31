# Natural Language Query Interface for JGI Integration Analysis

This document explains how to use the AI-powered query interface to explore your analysis data using natural language.

## Quick Start

After running your analysis, register your data and start querying:

```python
# Register all analysis data to the query database
analysis.register_all_existing_data()

# Query using natural language
result = analysis.query("Show all rows from the feature network node table where the node is in submodule 1")
display(result)
```

## Available Data Tables

The query interface has access to several key data tables:

### Analysis Tables
- **`feature_network_node_table`** - Network nodes with submodule assignments and annotations
- **`feature_correlation_table`** - Pairwise feature correlations 
- **`feature_annotation_table`** - Feature annotations across all datasets
- **`integrated_data`** - Combined data matrix from all datasets
- **`integrated_metadata`** - Unified metadata across samples
- **`functional_enrichment_table`** - Functional enrichment results by submodule

### Dataset Tables (per dataset type: mx, tx, etc.)
- **`raw_data`** - Original unprocessed data
- **`normalized_data`** - Final processed data ready for analysis
- **`linked_metadata`** - Sample metadata linked across datasets
- **`annotation_table`** - Feature ID to annotation mappings

## Query Writing Best Practices

### 1. Use Clear Table References

**Good:**
```python
analysis.query("Show all rows from the feature network node table")
analysis.query("Get data from the feature correlation table")
```

**Avoid ambiguous references:**
```python
analysis.query("Show me the data")  # Too vague
```

### 2. Specify Filters Clearly

**Submodule Filtering:**
```python
# Submodules are stored as strings like 'submodule_1', 'submodule_15'
analysis.query("Show nodes from the feature network node table where submodule is 1")
analysis.query("Find all features in submodule 15")
```

**Annotation Filtering:**
```python
analysis.query("Show nodes where mx_npclassifier_superclass is not 'Unassigned'")
analysis.query("Find genes where tx_cog_category contains 'metabolism'")
```

**Correlation Filtering:**
```python
analysis.query("Show correlations from the feature correlation table where correlation > 0.8")
analysis.query("Find negative correlations below -0.6")
```

### 3. Combine Multiple Conditions

**Good combinations:**
```python
analysis.query("Show nodes from the feature network node table where submodule is 1 and mx_npclassifier_superclass is not 'Unassigned'")

analysis.query("Find correlations where feature1 starts with 'mx_' and feature2 starts with 'tx_' and correlation > 0.7")

analysis.query("Show metadata where treatment is 'high_temp' and time_point > 24")
```

### 4. Use Proper Column Names

Common column patterns to reference:

**Network Node Table:**
- `feature_id` - The feature identifier
- `submodule` - Network submodule assignment (e.g., 'submodule_1')
- `dataset` - Dataset origin (mx, tx, etc.)
- `*_annotation_columns` - Various annotation fields

**Correlation Table:**
- `feature1`, `feature2` - The correlated feature pair
- `correlation` - Correlation coefficient
- `p_value` - Statistical significance

**Metadata Tables:**
- `sample_id` - Sample identifier
- `treatment`, `time_point`, `group` - Common experimental variables

### 5. Handle String Matching

**Exact matches:**
```python
analysis.query("Show nodes where dataset = 'mx'")
```

**Pattern matching:**
```python
analysis.query("Find features where feature_id starts with 'mx_'")
analysis.query("Show annotations where description contains 'amino acid'")
```

**Exclusions:**
```python
analysis.query("Show nodes where annotation is not 'Unknown'")
analysis.query("Find correlations where feature1 does not start with 'tx_'")
```

## Example Queries by Use Case

### Exploring Network Submodules

```python
# Get all features in a specific submodule
analysis.query("Show all nodes from the feature network node table where submodule is 5")

# Find submodules with metabolites
analysis.query("Show distinct submodule from the feature network node table where dataset = 'mx'")

# Get annotated features in submodules
analysis.query("Show nodes where submodule is 3 and mx_npclassifier_pathway is not 'Unassigned'")
```

### Correlation Analysis

```python
# Find strong positive correlations
analysis.query("Show correlations from the feature correlation table where correlation > 0.8")

# Cross-dataset correlations
analysis.query("Find correlations where feature1 starts with 'mx_' and feature2 starts with 'tx_'")

# Significant correlations
analysis.query("Show correlations where p_value < 0.001 and correlation > 0.6")
```

### Functional Analysis

```python
# Find enriched pathways
analysis.query("Show functional enrichment where p_value < 0.05")

# Submodule-specific enrichment
analysis.query("Find functions enriched in submodule 2 where corrected p value is less than 0.1")
```

### Metadata Exploration

```python
# Sample filtering
analysis.query("Show metadata where treatment is 'control' and time_point is 0")

# Count samples by condition
analysis.query("Show treatment, count(*) from integrated_metadata group by treatment")
```

## Tips for Better Results

### 1. Be Specific About What You Want
- Instead of "show data", say "show all rows from the feature network node table"
- Specify exact column names when possible

### 2. Use Standard SQL-like Language
- "where", "and", "or", "not", "in", "like", "contains"
- "greater than" or ">", "less than" or "<"
- "starts with", "ends with", "contains"

### 3. Check Available Tables First
```python
# See what tables are available
tables = analysis.query("show tables")
display(tables)
```

### 4. Handle Empty Results
```python
result = analysis.query("your query here")
if result.empty:
    print("No results found - try adjusting your query")
else:
    display(result)
```

### 5. Use Limits for Large Results
```python
# The system automatically adds LIMIT 100, but you can be explicit
analysis.query("Show nodes from feature network node table limit 50")
```

## Troubleshooting

### Common Issues

1. **"Query pattern not recognized"** - Be more specific about table names
2. **Empty results** - Check column names and values
3. **SQL syntax errors** - Use simpler language, avoid complex nested queries

### Getting Help

```python
# See available data
schema = analysis.data_registry.get_schema_info()
print(schema)

# Check what's in a specific table
sample = analysis.query("Show * from feature_network_node_table limit 5")
print(sample.columns.tolist())
```

## Advanced Examples

```python
# Complex filtering with multiple conditions
metabolites_in_submod = analysis.query("Show nodes from feature network node table where dataset = 'mx' and submodule in (1, 2, 3) and mx_npclassifier_superclass is not 'Unassigned'")

# Cross-table analysis (when supported)
strong_correlations = analysis.query("Show feature1, feature2, correlation from feature correlation table where correlation > 0.8 order by correlation desc")
```

This interface makes it easy to explore your multi-omics data without writing complex pandas operations!