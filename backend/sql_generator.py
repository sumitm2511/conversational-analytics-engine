import os
import re
from groq import Groq

class SQLGenerator:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_sql(self, question: str, enriched_schema: str, target_tables: list = None) -> str:
        table_hint = ""
        if target_tables:
            table_hint = f"\nThe primary tables to query are: {', '.join(target_tables)}\nIf querying multiple tables, make sure to write an appropriate JOIN using the relationships mapped below."

        prompt = f"""
        You are an expert SQL analyst. Given the schema below, write a single 
        DuckDB-compatible SQL query that answers the user's question.

        Rules:
        - Return ONLY the raw SQL query — no explanation, no markdown, no fences
        - Use ONLY table and column names that exist in the schema
        - Use readable column aliases (e.g. AS total_sales)
        - For date/time, use DuckDB syntax: DATE_TRUNC, EXTRACT, strftime
        - For top-N use ORDER BY ... DESC LIMIT N
        - Prefer CTEs over nested subqueries for readability
        - If aggregating, always GROUP BY all non-aggregated columns
        - Never use SELECT * — always name the columns you need
        {table_hint}

        Schema & Relationships:
        {enriched_schema}

        Question: {question}
        """
        
        resp = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        sql = resp.choices[0].message.content.strip()
        
        # Safely clean up markdown formatting without using literal backticks in the code
        backticks = chr(96) * 3
        sql = sql.replace(backticks + "sql", "").replace(backticks, "").strip()
        
        return sql

    def fix_sql(self, original_sql: str, error: str, enriched_schema: str) -> str:
        """Ask the LLM to fix a broken SQL query."""
        prompt = f"""
        You are a DuckDB SQL expert. The following query produced an error.
        Fix it and return ONLY the corrected SQL — no explanation, no markdown.

        Schema & Relationships:
        {enriched_schema}

        Broken SQL:
        {original_sql}

        Error:
        {error}
        """
        
        resp = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        sql = resp.choices[0].message.content.strip()
        
        # Safely clean up markdown formatting without using literal backticks in the code
        backticks = chr(96) * 3
        sql = sql.replace(backticks + "sql", "").replace(backticks, "").strip()
        
        return sql