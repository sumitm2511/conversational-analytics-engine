import duckdb
import pandas as pd
import re

class AnalyticsEngine:

    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self.tables = {}
        self.dataframes = {}   # raw DFs for forecasting / trend engines
        self.active_table = None

    # ── helpers ────────────────────────────────────────────────────────

    def clean_column(self, col: str) -> str:
        col = str(col).strip()
        col = re.sub(r"[^\w]+", "_", col)
        return col.lower()

    def make_unique(self, cols: list) -> list:
        seen, final = {}, []
        for col in cols:
            if col in seen:
                seen[col] += 1
                final.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final.append(col)
        return final

    def _classify_dtype(self, series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if series.dtype == object:
            sample = series.dropna().head(50)
            try:
                pd.to_datetime(sample)
                return "date"
            except Exception:
                pass
        return "text"

    def _try_parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(100)
                try:
                    parsed = pd.to_datetime(sample)
                    if parsed.notna().sum() > len(sample) * 0.8:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
        return df

    def _register(self, table_name: str, df: pd.DataFrame):
        df = self._try_parse_dates(df)
        df = df.copy()  # 🔥 CRITICAL: release file lock

        self.conn.register(table_name, df)
        self.dataframes[table_name] = df

        self.tables[table_name] = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "dtypes": {col: self._classify_dtype(df[col]) for col in df.columns},
            "raw_dtypes": {col: str(df[col].dtype) for col in df.columns},
            # Computed per-table now, so every sheet in a multi-sheet upload
            # gets its own real health score instead of only the active table.
            "health": self.calculate_data_health(df),
        }

    # ── Data Health Scanner ──────────────────────────────────────────────

    def calculate_data_health(self, df: pd.DataFrame) -> dict:
        """Scans the dataframe for missing values and duplicates to generate a health score.

        Score weights nulls more heavily than duplicates (0.7 / 0.3 split) since
        missing values directly break downstream forecast/regression/cluster
        engines, while duplicate rows are a data-quality smell that's usually
        less immediately harmful but still worth surfacing and penalizing.
        """
        total_rows = len(df)
        if total_rows == 0:
            return {"score": 0, "warnings": ["Dataset is entirely empty."]}

        total_cells    = total_rows * len(df.columns)
        missing_cells  = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        null_pct = missing_cells / total_cells if total_cells else 0
        dup_pct  = duplicate_rows / total_rows if total_rows else 0

        health_score = max(
            0,
            round(100 * (1 - null_pct * 0.7 - dup_pct * 0.3), 1)
        )

        warnings = []
        if missing_cells > 0:
            warnings.append(
                f"Found {missing_cells} missing values. "
                "(Our AI engines will automatically interpolate these during analysis)."
            )
        if duplicate_rows > 0:
            warnings.append(
                f"Found {duplicate_rows} duplicate rows. Consider cleaning your source file."
            )

        return {
            "score": health_score,
            "warnings": warnings,
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
        }

    # ── public API ─────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> dict:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            df.columns = self.make_unique([self.clean_column(c) for c in df.columns])
            df = df.copy()
            self._register("sheet1", df)
        else:
            with pd.ExcelFile(file_path) as excel:
                for sheet in excel.sheet_names:
                    df = excel.parse(sheet)
                    df.columns = self.make_unique([self.clean_column(c) for c in df.columns])
                    df = df.copy()
                    table_name = re.sub(r"[^\w]+", "_", sheet.lower())
                    self._register(table_name, df)

        self.active_table = list(self.tables.keys())[0]

        return {
            "tables": list(self.tables.keys()),
            "active_table": self.active_table,
            "columns": self.tables[self.active_table]["columns"],
            # Each entry in table_details now carries its own "health" dict
            # (score + warnings + missing_cells + duplicate_rows), computed
            # per-sheet in _register() rather than once for the active table only.
            "table_details": self.tables,
        }

    def get_dataframe(self, table_name: str = None) -> pd.DataFrame:
        name = table_name or self.active_table
        return self.dataframes.get(name)

    def get_all_schemas(self) -> str:
        text = ""
        for table, info in self.tables.items():
            text += f"\nTable: {table}\nColumns:\n"
            for col in info["columns"]:
                dtype = info["dtypes"].get(col, "")
                text += f"  - {col} ({dtype})\n"
        return text

    def get_enriched_schema(self) -> str:
        """Generates schema info with explicitly detected join relationships."""
        if not self.tables:
            return "No tables loaded."

        schema_text = "Tables:\n"
        column_map = {}

        for table, info in self.tables.items():
            schema_text += f"\nTable: {table}\nColumns:\n"
            for col in info["columns"]:
                dtype = info["dtypes"].get(col, "")
                schema_text += f"  - {col} ({dtype})\n"

                # Map column to tables for relationship detection
                if col not in column_map:
                    column_map[col] = []
                column_map[col].append(table)

        # Detect possible joins (columns that exist in >1 table)
        # We ignore generic common names that usually aren't keys
        ignore_cols = {'id', 'date', 'created_at', 'updated_at', 'index', 'name', 'status', 'type'}
        possible_joins = {
            col: tbls for col, tbls in column_map.items()
            if len(tbls) > 1 and col.lower() not in ignore_cols
        }

        if possible_joins:
            schema_text += "\nPossible Join Relationships (Foreign Keys):\n"
            for col, tbls in possible_joins.items():
                schema_text += f"- Column '{col}' can link tables: {', '.join(tbls)}\n"

        return schema_text

    def execute_sql(self, sql: str) -> dict:
        try:
            df = self.conn.execute(sql).fetchdf()
            return {"data": df, "error": None}
        except Exception as e:
            return {"data": None, "error": str(e)}