# What Unity Catalog actually changes

Unity Catalog is a catalogue over the tables in a workspace: it records which principal may read which table, and it keeps lineage from a dashboard back to the columns it reads. On Databricks that replaces per-table grants scattered across notebooks, and it is what makes a table safe to share outside the team that wrote it.
