# migrate_data_production.py
import pandas as pd
from sqlalchemy import create_engine, inspect

# Configuration for the old and new databases
OLD_DB_URL = "postgresql://postgres.njsusvrlnigkiduefyax:GfSNK6BnVuBUYEf7@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
NEW_DB_URL = "postgresql://new_user:new_password@new_host:5432/new_db"
COMPANY = "Electricity Company of Ghana Ltd"
ORG_TABLE = "Organization"

def get_engine(db_url):
    return create_engine(db_url)

def get_inspector(engine):
    return inspect(engine)

def load_table_data(table, engine):
    query = f'SELECT * FROM "{table}"'
    return pd.read_sql(query, engine)

def build_reverse_graph(engine):
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
    df = load_table_data(table, engine)
    if df.empty:
        return df
    if table == ORG_TABLE:
        return df[df["name"] == COMPANY]
    
    mask = pd.Series([False] * len(df))
    found_fk = False
    for fk in fk_info_by_child.get(table, []):
        parent_table = fk.get("referred_table")
        if parent_table in filtered_data:
            local_cols = fk.get("constrained_columns")
            referred_cols = fk.get("referred_columns")
            if not local_cols or not referred_cols:
                continue
            local_col = local_cols[0]
            referred_col = referred_cols[0]
            parent_df = filtered_data[parent_table]
            allowed_values = set(parent_df[referred_col].dropna().unique())
            if allowed_values:
                mask = mask | df[local_col].isin(allowed_values)
                found_fk = True
    if not found_fk:
        return df.iloc[0:0]
    return df[mask]

def migrate_data_to_production():
    old_engine = get_engine(OLD_DB_URL)
    new_engine = get_engine(NEW_DB_URL)
    
    reverse_graph, fk_info_by_child = build_reverse_graph(old_engine)
    related_tables = get_related_tables(old_engine, ORG_TABLE)
    sorted_tables = topological_sort_tables(old_engine, related_tables, reverse_graph)
    print("Processing tables in topological order:", sorted_tables)
    
    filtered_data = {}
    
    for table in sorted_tables:
        df_filtered = filter_table_data(table, old_engine, filtered_data, fk_info_by_child)
        if df_filtered.empty:
            print(f"Skipping table '{table}' (no related data).")
        else:
            filtered_data[table] = df_filtered
            # Write the filtered data to the new production database.
            df_filtered.to_sql(table, new_engine, if_exists="replace", index=False)
            print(f"Table '{table}' migrated successfully.")
    print("Final production migration for Electricity Company of Ghana Ltd is complete.")

if __name__ == "__main__":
    migrate_data_to_production()
