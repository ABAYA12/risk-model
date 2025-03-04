# show_tables.py
from sqlalchemy import create_engine, inspect

# Database connection string (update if needed)
DB_URL =  "postgresql://postgres.njsusvrlnigkiduefyax:GfSNK6BnVuBUYEf7@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
COMPANY = "Electricity Company of Ghana Ltd"

def list_tables_for_company():
    engine = create_engine(DB_URL)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    output_lines = []
    output_lines.append("Tables in the database relevant to Electricity Company of Ghana Ltd:")
    
    for table in tables:
        columns = inspector.get_columns(table)
        col_names = [col["name"] for col in columns]
        
        # Check if table is company-specific by looking for a company column.
        if "company_name" in col_names or "company" in col_names:
            # Build a query to check if this table has data for our company.
            if "company_name" in col_names:
                query = f'SELECT 1 FROM "{table}" WHERE "company_name" = %s LIMIT 1;'
            else:
                query = f'SELECT 1 FROM "{table}" WHERE "company" = %s LIMIT 1;'
            result = engine.execute(query, (COMPANY,))
            if result.fetchone():
                output_lines.append(f" - {table} (company-specific)")
        else:
            # Table doesn't have a company column – assume it's shared data.
            output_lines.append(f" - {table} (shared or non-specific)")
    
    # Write the output to a text file
    with open("tables_list.txt", "w") as file:
        file.write("\n".join(output_lines))
    
    print("Table list saved to tables_list.txt")

if __name__ == "__main__":
    list_tables_for_company()
