# convert_tables_to_json_separate.py
import os
import json
import pandas as pd
from sqlalchemy import create_engine, inspect

# Configuration for the old database and target company
OLD_DB_URL = "postgresql://postgres.njsusvrlnigkiduefyax:password@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
COMPANY = "Electricity Company of Ghana Ltd"
ORG_TABLE = "Organization"  # Table names are case sensitive

def get_engine():
    return create_engine(OLD_DB_URL)

def get_inspector(engine):
    return inspect(engine)

def load_table_data(table, engine):
    # Use double quotes to preserve uppercase table names
    query = f'SELECT * FROM "{table}"'
    return pd.read_sql(query, engine)

def build_reverse_graph(engine):
    """
    Build a reverse mapping (parent -> list of (child, fk_info)) for all tables,
    and return a dictionary mapping each child table to its foreign key definitions.
    """
    inspector = get_inspector(engine)
    tables = inspector.get_table_names()
    reverse_graph = {table: [] for table in tables}
    fk_info_by_child = {}
    for table in tables:
        fks = inspector.get_foreign_keys(table)
        fk_info_by_child[table] = fks
        for fk in fks:
            parent = fk.get("referred_table")
            if parent in reverse_graph:
                reverse_graph[parent].append((table, fk))
    return reverse_graph, fk_info_by_child

def get_related_tables(engine, org_table=ORG_TABLE):
    """
    Traverse the reverse graph starting from the Organization table to
    collect all tables that are directly or indirectly related.
    """
    reverse_graph, _ = build_reverse_graph(engine)
    related = set()
    stack = [org_table]
    while stack:
        current = stack.pop()
        if current not in related:
            related.add(current)
            for child, _ in reverse_graph.get(current, []):
                stack.append(child)
    return related

def topological_sort_tables(engine, related_tables, reverse_graph):
    """
    Topologically sort the related tables so that parent tables are processed
    before their dependent child tables.
    """
    inspector = get_inspector(engine)
    in_degree = {table: 0 for table in related_tables}
    for table in related_tables:
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            parent = fk.get("referred_table")
            if parent in related_tables:
                in_degree[table] += 1
    sorted_tables = []
    queue = [table for table, degree in in_degree.items() if degree == 0]
    while queue:
        table = queue.pop(0)
        sorted_tables.append(table)
        for child, _ in reverse_graph.get(table, []):
            if child in in_degree:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    return sorted_tables

def filter_table_data(table, engine, filtered_data, fk_info_by_child):
    """
    Filter a given table’s data so that only rows connected (directly or indirectly)
    to the target organization are kept.
      - For the Organization table, filter on "name" == COMPANY.
      - For other tables, use any foreign key referencing a parent table that has already been filtered.
    """
    df = load_table_data(table, engine)
    if df.empty:
        return df

    if table == ORG_TABLE:
        return df[df["name"] == COMPANY]

    # Initialize a mask of False values
    mask = pd.Series([False] * len(df))
    found_fk = False
    for fk in fk_info_by_child.get(table, []):
        parent_table = fk.get("referred_table")
        if parent_table in filtered_data:
            local_cols = fk.get("constrained_columns")
            referred_cols = fk.get("referred_columns")
            if not local_cols or not referred_cols:
                continue
            # Assume a single-column foreign key
            local_col = local_cols[0]
            referred_col = referred_cols[0]
            parent_df = filtered_data[parent_table]
            allowed_values = set(parent_df[referred_col].dropna().unique())
            if allowed_values:
                mask = mask | df[local_col].isin(allowed_values)
                found_fk = True
    if not found_fk:
        # If no foreign key could be used for filtering, return an empty DataFrame.
        return df.iloc[0:0]
    return df[mask]

def export_related_tables_to_json():
    engine = get_engine()
    reverse_graph, fk_info_by_child = build_reverse_graph(engine)
    related_tables = get_related_tables(engine, ORG_TABLE)
    sorted_tables = topological_sort_tables(engine, related_tables, reverse_graph)
    print("Processing tables in topological order:", sorted_tables)
    
    filtered_data = {}  # Dictionary to store filtered DataFrames keyed by table name
    output_dir = "json_exports"
    os.makedirs(output_dir, exist_ok=True)
    
    for table in sorted_tables:
        df_filtered = filter_table_data(table, engine, filtered_data, fk_info_by_child)
        if df_filtered.empty:
            print(f"Skipping table '{table}' (no related data).")
        else:
            filtered_data[table] = df_filtered
            output_file = os.path.join(output_dir, f"{table}.json")
            with open(output_file, "w") as f:
                json.dump(json.loads(df_filtered.to_json(orient="records")), f, indent=4)
            print(f"Exported table '{table}' to {output_file}.")
    print("Export complete.")

if __name__ == "__main__":
    export_related_tables_to_json()
