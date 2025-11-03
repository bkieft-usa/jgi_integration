# Natural Language Query Interface for JGI Integration Analysis

This document explains how to use the AI-powered query interface to explore your analysis data using natural language. This interface makes it easier to explore your multi-omics data results without writing complex pandas operations!

## Quick Start

After running your analysis, register your data and start querying:

```python
# Deposit all analysis data to the query database
analysis.register_all_existing_data()

# Query using natural language
result = analysis.query("Show all nodes from the network that are in submodule 1")
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
analysis.query("Find all features that are annotated to a metabolite superclass, class, or subclass which includes the word 'Polyamines'")
analysis.query("Find functions enriched in any submodule where corrected p value is less than 0.1")
analysis.query("Find feature pairs with a correlation score that is at least 0.95")
```

**Avoid ambiguous references:**
```python
analysis.query("Show me the data tables")  # Too vague
analysis.query("What are my most important genes?")  # Ambiguous language
```

### 2. Specify Filters Clearly

**Submodule Filtering:**
```python
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

analysis.query("Show metadata where treatment is 'high_temp' and time_point > 24")
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
tables = analysis.query("show all data tables that are available to query")
display(tables)
```

### 4. Use Limits for Large Results
```python
# The system automatically adds LIMIT 500 as a guard against pulling very large dataframes, but you can be explicit
more_rows = analysis.query("Show nodes from feature network node table, limit 1000")
display(more_rows)
```

## Troubleshooting

### Common Issues

1. **"Query pattern not recognized"** - Be more specific about table names
2. **Empty results** - Check column names and values, but data may not exist
3. **SQL syntax errors** - Use simpler language, avoid complex nested queries

### Getting Help

```python
# See available data
schema = analysis.data_registry.get_schema_info()
print(schema)

# Check what's in a specific table
sample = analysis.query("Show * from feature_network_node_table, limit 5")
print(sample.columns.tolist())
```

## Advanced Examples

```python
# Complex filtering with multiple conditions
metabolites_in_submod = analysis.query("Show nodes from feature network node table where dataset = 'mx' and submodule in (1, 2, 3) and mx_npclassifier_superclass is not 'Unassigned'")

# Cross-table analysis (soon to be supported)
strong_correlations = analysis.query("Show feature1, feature2, correlation from feature correlation table where correlation > 0.8 order by correlation desc")
```