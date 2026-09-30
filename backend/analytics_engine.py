import duckdb
import pandas as pd
import re
from pathlib import Path

# Generic column names unlikely to represent a real relationship between two
# tables (an "id" or "status" column existing in both is usually coincidence,
# not a meaningful join key). Shared by get_enriched_schema()'s AI-facing join
# hints and suggest_joins()'s user-facing auto-join suggestions, so both stay
# in sync rather than drifting into two different ignore lists over time.
_IGNORE_JOIN_COLS = {'id', 'date', 'created_at', 'updated_at', 'index', 'name', 'status', 'type'}

# Shared SQL join keyword per join_type, matching the vocabulary
# IntentClassifier's join_type field and SQLGenerator's own mapping already
# use, so all three stay in sync on what "left"/"right"/"full" mean.
_JOIN_TYPE_SQL = {
    "inner": "JOIN",
    "left":  "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full":  "FULL OUTER JOIN",
}

# Common identifier/code-style column-name suffixes — deliberately broader
# than just "_id" (also catches invoice_no, product_code, account_key,
# etc.), since the underlying problem generalizes: any column that's
# really a label/reference, not a measurable quantity, shouldn't be
# summed/averaged no matter what it happens to be called.
_ID_NAME_PATTERN = re.compile(r'(^|_)(id|no|num|number|code|key)$')


def is_identifier_like(col_name: str, series: pd.Series) -> bool:
    """
    True when a column looks like an identifier/code/key rather than a
    measurable quantity — by name pattern, or by being almost entirely
    unique-valued (behaves like an identifier even under an unrelated
    name). This is the ONE place this decision gets made. Every consumer
    that needs to know "is it safe to sum/average this column" — the
    schema text shown to every LLM prompt (SQLGenerator, IntentClassifier,
    InsightEngine all read from get_all_schemas()/get_enriched_schema()),
    and main.py's extract_metrics() — calls this instead of each keeping
    its own copy of the same heuristic. That duplication is exactly what
    let a meaningless SUM(employee_id) get computed once in Python and
    then independently re-derived and mislabeled a second time by the LLM
    ("Total Employees: 1,379") — two places quietly disagreeing, the same
    failure mode the dtype-detection rewrite earlier in this codebase was
    built to eliminate.
    """
    name = str(col_name).lower()
    if _ID_NAME_PATTERN.search(name):
        return True

    n = len(series)
    if n > 5:
        try:
            if series.nunique(dropna=True) / n > 0.95:
                return True
        except Exception:
            pass

    return False

# Canonical boolean vocabulary used by _try_parse_boolean(). A column only
# gets coerced when its ENTIRE distinct value set is a subset of this (with
# at least one true-ish and one false-ish value present) — deliberately
# stricter than the majority-threshold used for dates/numbers, since a
# lossy true/false coercion on a column that also contains e.g. "pending"
# would silently discard that third value.
_BOOL_TRUE  = {"true", "yes", "y", "1"}
_BOOL_FALSE = {"false", "no", "n", "0"}


def _is_textual(series: pd.Series) -> bool:
    """
    True for a column still holding text, not yet coerced to something more
    specific. Checking `dtype == object` alone — the previous approach —
    silently breaks on pandas >= 3.0, which changed the DEFAULT dtype for
    plain string columns from "object" to a dedicated "str" (StringDtype).
    Under that pandas version, `dtype == object` is False for every text
    column, so every one of the coercion passes below would silently skip
    every column — reproducing exactly "everything shows as text no matter
    what", with no exception raised anywhere to reveal why.
    is_string_dtype() correctly recognizes both the legacy object-based
    string columns (older pandas) and the new native string dtype (pandas
    3.0+), so this check works across versions instead of assuming one.
    """
    return pd.api.types.is_string_dtype(series) or series.dtype == object


class AnalyticsEngine:

    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self.tables = {}
        self.dataframes = {}   # raw DFs for forecasting / trend engines
        self.active_table = None
        # Tracks table pairs that have already been combined this session,
        # as frozenset({table_a, table_b}) so order doesn't matter. Consulted
        # by suggest_joins() so a pair doesn't keep reappearing in the
        # "Combine tables" suggestions every time schema is refetched after
        # a combine (combine_tables() intentionally leaves the two source
        # tables loaded, so without this they'd still share columns and
        # would otherwise be suggested again indefinitely).
        self.combined_pairs = set()

    # ── helpers ────────────────────────────────────────────────────────

    def clean_column(self, col: str) -> str:
        col = str(col).strip()
        col = re.sub(r"[^\w]+", "_", col)
        return col.lower()

    def _unique_table_name(self, base: str) -> str:
        """
        Guarantees a table name doesn't collide with one already loaded this
        session, appending _2, _3, ... as needed. Without this, two uploads
        that would otherwise land on the same name (two CSVs — previously
        BOTH always named "sheet1" regardless of filename — or two Excel
        files that each happen to have a sheet literally named "Sheet1")
        silently overwrote each other in self.tables and in DuckDB, instead
        of coexisting as two separate tables.
        """
        name = base or "table"
        i = 2
        while name in self.tables:
            name = f"{base}_{i}"
            i += 1
        return name

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

    def _classify_dtype(self, series: pd.Series, is_percent: bool = False) -> str:
        """
        Reads off the FINAL semantic category for an already-coerced column.
        Deliberately does NOT do any of its own independent parsing/guessing
        anymore — that used to live here AND in _try_parse_dates() as two
        separate heuristics with different thresholds, which could (and did)
        disagree with each other on the same column. Now there is exactly
        ONE place that decides a column's type (the _try_parse_* methods,
        run once in _register()), and this function just labels the result.
        """
        if is_percent:
            return "percent"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "text"

    def _try_parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if _is_textual(df[col]):
                sample = df[col].dropna().head(100)
                try:
                    parsed = pd.to_datetime(sample)
                    if parsed.notna().sum() > len(sample) * 0.8:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
        return df

    def _try_parse_boolean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerces an object column to real boolean dtype when its full set of
        distinct non-null values (trimmed, case-insensitive) is a subset of
        the canonical yes/no vocabulary above, with at least one value from
        each side present. Runs BEFORE numeric coercion so a genuine
        yes/no/true/false flag column — including one encoded as literal
        "0"/"1" strings — is recognized as boolean rather than being caught
        by the more permissive numeric coercion first.
        """
        for col in df.columns:
            if not _is_textual(df[col]):
                continue

            non_null = df[col].dropna().astype(str).str.strip().str.lower()
            if non_null.empty:
                continue

            distinct = set(non_null.unique())
            if distinct <= (_BOOL_TRUE | _BOOL_FALSE) and (distinct & _BOOL_TRUE) and (distinct & _BOOL_FALSE):
                df[col] = df[col].astype(str).str.strip().str.lower().map(
                    lambda v: True if v in _BOOL_TRUE else (False if v in _BOOL_FALSE else None)
                )
        return df

    def _try_parse_numeric(self, df: pd.DataFrame) -> tuple:
        """
        For any column still 'object' after date/boolean parsing, tries to
        coerce it to a real numeric dtype — stripping common formatting
        (currency symbols, thousands separators, percent signs, surrounding
        whitespace) rather than only accepting bare numbers, since Excel/CSV
        exports of things like revenue ("$1,234.56") or rates ("45%") land
        as plain text otherwise. Uses the same majority-threshold approach
        as _try_parse_dates: only commits the conversion if over 80% of the
        column's non-null values parse cleanly, so a genuinely text column
        with a few numeric-looking values isn't wrongly coerced.

        Returns (df, percent_cols) — percent_cols is the set of column names
        that were percent-formatted (e.g. "45%"), tracked separately since
        after conversion their pandas dtype (float64) is indistinguishable
        from any other numeric column; _register() uses this set to label
        them "percent" instead of plain "numeric".
        """
        percent_cols = set()

        for col in df.columns:
            if not _is_textual(df[col]):
                continue

            non_null_original = df[col].notna().sum()
            if non_null_original == 0:
                continue

            str_col = df[col].astype(str).str.strip()
            is_percent = str_col[df[col].notna()].str.endswith("%").mean() > 0.8

            cleaned = (
                str_col
                .str.replace(r"[,$€£]", "", regex=True)
                .str.replace("%", "", regex=False)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")

            success_rate = parsed.notna().sum() / non_null_original
            if success_rate > 0.8:
                if is_percent:
                    parsed = parsed / 100.0
                    percent_cols.add(col)
                df[col] = parsed

        return df, percent_cols

    def _register(self, table_name: str, df: pd.DataFrame):
        # Captured BEFORE any coercion so we can tell, per column, whether a
        # parsing step turned a previously-present value into null — that's
        # data loss worth surfacing explicitly rather than letting it vanish
        # silently into the general null count.
        original_nulls = df.isnull().sum()

        df = self._try_parse_dates(df)
        df = self._try_parse_boolean(df)
        df, percent_cols = self._try_parse_numeric(df)
        df = df.copy()  # 🔥 CRITICAL: release file lock

        self.conn.register(table_name, df)
        self.dataframes[table_name] = df

        dtypes = {
            col: self._classify_dtype(df[col], is_percent=(col in percent_cols))
            for col in df.columns
        }

        # Aggregation role per column — computed once, here, using the
        # shared is_identifier_like() check. "identifier" columns are
        # excluded from any sum/average anywhere downstream (see
        # get_all_schemas()/get_enriched_schema() annotating them for every
        # LLM prompt, and main.py's extract_metrics() skipping them for
        # Python-computed KPIs).
        roles = {}
        for col in df.columns:
            dt = dtypes[col]
            if dt == "percent":
                roles[col] = "percentage"
            elif dt == "numeric":
                roles[col] = "identifier" if is_identifier_like(col, df[col]) else "measure"
            elif dt == "text":
                roles[col] = "categorical"
            else:
                roles[col] = "other"

        # Computed per-table now, so every sheet in a multi-sheet upload
        # gets its own real health score instead of only the active table.
        health = self.calculate_data_health(df)

        new_nulls = df.isnull().sum()
        for col in df.columns:
            added = int(new_nulls.get(col, 0)) - int(original_nulls.get(col, 0))
            if added > 0:
                health["warnings"].append(
                    f"Column '{col}': {added} value(s) couldn't be parsed as "
                    f"{dtypes[col]} and were treated as missing."
                )

        self.tables[table_name] = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "dtypes": dtypes,
            "roles": roles,
            "raw_dtypes": {col: str(df[col].dtype) for col in df.columns},
            "health": health,
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

    def inspect_excel_sheets(self, file_path: str) -> list:
        """
        Returns just the sheet names in an Excel file, without loading any
        data or touching session state. Used by the /inspect_file route so
        the frontend can show a sheet picker BEFORE committing to loading
        (potentially large) data for sheets the user doesn't want.
        """
        with pd.ExcelFile(file_path) as excel:
            return list(excel.sheet_names)

    def load_file(self, file_path: str, original_filename: str = None, sheets: list = None) -> dict:
        """
        Loads a CSV or Excel file.

        `original_filename`: the name the user's file had before it was
        saved to a temp path (main.py's /upload route saves every upload to
        a randomly-named temp file, so the real name has to be passed
        through explicitly — file_path alone is useless for naming). Used
        as the base table name for a CSV upload. If omitted, falls back to
        the old fixed name "sheet1" for backward compatibility with any
        existing caller that doesn't pass it — but note that means the
        original collision bug (every CSV overwriting the last) comes back
        for that caller, so any route that accepts uploads should pass this.

        `sheets`, Excel only: if given, only these sheet names are loaded —
        each becomes its own table, exactly as before, just filtered to the
        user's selection instead of automatically loading every sheet in the
        workbook. If omitted (None), every sheet is loaded, same as the
        original behavior, so this stays backward compatible with any
        existing caller that doesn't pass it.

        Table names are deduplicated via _unique_table_name() against
        whatever's already loaded this session — this is what lets a second
        CSV (or an Excel sheet sharing a name with one already loaded)
        become its own table instead of silently overwriting an earlier one.
        """
        newly_loaded = []

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            df.columns = self.make_unique([self.clean_column(c) for c in df.columns])
            df = df.copy()
            base_name = (
                self.clean_column(Path(original_filename).stem)
                if original_filename else "sheet1"
            )
            table_name = self._unique_table_name(base_name)
            self._register(table_name, df)
            newly_loaded.append(table_name)
        else:
            with pd.ExcelFile(file_path) as excel:
                available = excel.sheet_names
                sheet_names = sheets if sheets else available

                # Defensive: ignore any requested sheet name that doesn't
                # actually exist in this workbook rather than raising, since
                # a stale selection (e.g. file re-picked after a sheet was
                # renamed) shouldn't hard-fail the whole upload.
                sheet_names = [s for s in sheet_names if s in available]
                if not sheet_names:
                    raise ValueError(
                        f"None of the requested sheets {sheets!r} exist in this file. "
                        f"Available sheets: {available}"
                    )

                for sheet in sheet_names:
                    df = excel.parse(sheet)
                    df.columns = self.make_unique([self.clean_column(c) for c in df.columns])
                    df = df.copy()
                    base_name = re.sub(r"[^\w]+", "_", sheet.lower())
                    table_name = self._unique_table_name(base_name)
                    self._register(table_name, df)
                    newly_loaded.append(table_name)

        # Active table becomes the first sheet loaded THIS call, not just
        # "whatever happens to be first in self.tables overall" — matters
        # once sheet selection means a fresh upload might not touch the
        # table that happened to load first in an earlier call this session.
        self.active_table = newly_loaded[0]

        return {
            "tables": list(self.tables.keys()),
            "active_table": self.active_table,
            "columns": self.tables[self.active_table]["columns"],
            # Each entry in table_details now carries its own "health" dict
            # (score + warnings + missing_cells + duplicate_rows), computed
            # per-sheet in _register() rather than once for the active table only.
            "table_details": self.tables,
        }

    def _register_dataframe(self, df: pd.DataFrame, table_name: str) -> dict:
        """
        Shared tail-end of every "I have a DataFrame from an external source,
        register it as a table" path — file upload, MySQL, Snowflake, or
        anything added later. Column cleanup + _register() (schema detection,
        health scoring, DuckDB registration) + active_table bookkeeping all
        happen exactly once here, so every SQL-source connector behaves
        identically regardless of which database it queried.

        Note: does NOT clear previously-loaded tables — self.tables is
        cumulative across calls within a session, which is what makes
        multi-table loading (and combine_tables() below) work: loading
        table A, then table B, leaves both registered simultaneously. This
        also means tables from different origins (file, MySQL, Snowflake)
        sit in the exact same dict, with no origin tag distinguishing them —
        so suggest_joins()/combine_tables()/get_joined_dataframe() treat them
        identically and a table from any source can join/combine with a
        table from any other source with no extra work.
        """
        df.columns = self.make_unique([self.clean_column(c) for c in df.columns])
        df = df.copy()
        self._register(table_name, df)

        # Newly loaded data becomes the active table, same as a fresh file upload.
        self.active_table = table_name

        return {
            "tables": list(self.tables.keys()),
            "active_table": self.active_table,
            "columns": self.tables[self.active_table]["columns"],
            "table_details": self.tables,
        }

    def load_from_mysql(self, query: str, table_name: str = "mysql_data") -> dict:
        """
        Runs `query` against MySQL (via db_connectors.MySQLConnector) and
        registers the result via _register_dataframe() — see that method's
        docstring for what "registering" means and why origin doesn't matter
        downstream. Kept as its own named method (rather than folded away)
        since callers/tests may already reference it by this name.
        """
        from db_connectors import MySQLConnector
        df = MySQLConnector().run_query(query)
        return self._register_dataframe(df, table_name)

    def load_from_snowflake(self, query: str, table_name: str = "snowflake_data") -> dict:
        """
        Runs `query` against Snowflake (via db_connectors.SnowflakeConnector)
        and registers the result exactly like load_from_mysql() does for
        MySQL — same _register_dataframe() tail, so a Snowflake table is
        indistinguishable from a MySQL or uploaded-file table to every other
        part of this engine (joins, combine, health scoring, dtype detection).
        """
        from db_connectors import SnowflakeConnector
        df = SnowflakeConnector().run_query(query)
        return self._register_dataframe(df, table_name)

    def _shared_columns(self, table_a: str, table_b: str) -> list:
        """
        Non-generic column names shared between two tables — the same
        detection used by suggest_joins() for the UI's suggestion cards,
        extracted here so get_joined_dataframe() below relies on the exact
        same logic rather than a second copy that could quietly diverge.
        """
        cols_a = set(self.tables[table_a]["columns"])
        cols_b = set(self.tables[table_b]["columns"])
        return sorted(c for c in (cols_a & cols_b) if c.lower() not in _IGNORE_JOIN_COLS)

    def _build_join_sql(self, table_a: str, table_b: str, on_columns: list, join_keyword: str) -> str:
        """
        Shared SQL-building logic for combine_tables() and
        get_joined_dataframe() — both need identical column-collision
        handling (a non-join column present in both tables gets table_b's
        version suffixed "__<table_b>" rather than silently overwritten),
        kept in one place so the two methods can't drift apart on this.
        """
        cols_a = self.tables[table_a]["columns"]
        cols_b = self.tables[table_b]["columns"]

        select_parts = [f'a."{c}"' for c in cols_a]
        for c in cols_b:
            if c in on_columns:
                continue   # already included once via table_a
            if c in cols_a:
                select_parts.append(f'b."{c}" AS "{c}__{table_b}"')
            else:
                select_parts.append(f'b."{c}"')

        select_clause = ", ".join(select_parts)
        on_clause = " AND ".join(f'a."{c}" = b."{c}"' for c in on_columns)
        return (
            f'SELECT {select_clause} FROM "{table_a}" a '
            f'{join_keyword} "{table_b}" b ON {on_clause}'
        )

    def get_shared_columns(self, table_a: str, table_b: str) -> dict:
        """
        Public wrapper around _shared_columns() for the join-picker UI —
        returns every non-generic column shared between two tables,
        regardless of combined_pairs history (unlike suggest_joins(), which
        deliberately hides pairs already combined). The picker needs to
        offer a join between two tables even if they were combined before,
        since a live join is independent of any prior manual Combine.
        """
        missing = [t for t in (table_a, table_b) if t not in self.tables]
        if missing:
            return {"columns": [], "error": f"Unknown table(s): {missing}"}
        return {"columns": self._shared_columns(table_a, table_b), "error": None}

    def suggest_joins(self) -> list:
        """
        Detects table PAIRS that share one or more non-generic column names,
        and returns ONE suggestion per pair — not one per shared column.

        When two tables share several columns (e.g. market_area, country,
        commodity, product_area), they're almost always meant to be matched
        on the COMBINATION of those columns, not any single one in
        isolation: joining on just "market_area" while ignoring that
        "country"/"commodity" also need to match would create a many-to-many
        blowup — duplicated, incorrect rows — not a clean combine. So every
        shared column returned here is intended to be used together as a
        composite join key by combine_tables() below, whether there's one
        shared column or several.

        Works for any number of loaded tables (not just exactly two) by
        checking every pairwise combination — no n-way ambiguity to handle,
        since a "pair" is always just two tables regardless of how many are
        loaded overall. With 3+ tables loaded, this naturally returns
        multiple suggestion cards at once (one per qualifying pair) — there
        is no cap limiting it to a single suggestion at a time.

        Pairs already present in self.combined_pairs are skipped — otherwise,
        since combine_tables() intentionally leaves both source tables
        loaded (it only adds the new joined table), the exact same pair
        would keep reappearing here every time the frontend refetches
        suggestions after a combine.

        Returns: [{"table_a": ..., "table_b": ..., "columns": [...]}, ...]
        """
        from itertools import combinations

        suggestions = []
        for table_a, table_b in combinations(self.tables.keys(), 2):
            if frozenset({table_a, table_b}) in self.combined_pairs:
                continue

            shared = self._shared_columns(table_a, table_b)
            if shared:
                suggestions.append({"table_a": table_a, "table_b": table_b, "columns": shared})

        return suggestions

    def combine_tables(self, table_a: str, table_b: str, on_columns: list, new_table_name: str = None) -> dict:
        """
        Builds and runs an INNER JOIN between two already-loaded tables,
        matching on ALL of `on_columns` together (composite key) — see
        suggest_joins() docstring for why this matters for correctness when
        multiple columns are shared, not just convenience. Registers the
        result as a new table via the same _register() pipeline every other
        data source uses. No SQL is ever exposed to the caller.

        Column collisions (a non-join column with the same name in both
        tables) are resolved by suffixing table_b's version with
        "__<table_b>" rather than silently overwriting one.

        Works identically regardless of whether table_a/table_b came from a
        file upload or a MySQL connection — combine has no notion of source,
        only of table names already present in self.tables.

        Only ever combines exactly two tables per call. To combine three
        (or more) tables, call this twice in sequence: combine A+B into a
        new table, then combine THAT new table with C — since the combined
        table is registered exactly like any other, suggest_joins() will
        naturally offer it for combining with C if they share a column, and
        this method doesn't care that one side is itself a prior join
        result. A true single-call N-way join isn't implemented here.
        """
        missing = [t for t in (table_a, table_b) if t not in self.tables]
        if missing:
            return {"error": f"Unknown table(s): {missing}"}

        if not on_columns:
            return {"error": "At least one join column is required."}

        cols_a = self.tables[table_a]["columns"]
        cols_b = self.tables[table_b]["columns"]

        bad_cols = [c for c in on_columns if c not in cols_a or c not in cols_b]
        if bad_cols:
            return {"error": f"Column(s) not found in both '{table_a}' and '{table_b}': {bad_cols}"}

        if not new_table_name:
            new_table_name = f"{table_a}_{table_b}_combined"
        new_table_name = self.clean_column(new_table_name)

        sql = self._build_join_sql(table_a, table_b, on_columns, join_keyword="JOIN")

        result = self.execute_sql(sql)
        if result["error"]:
            return {"error": result["error"]}

        df = result["data"]
        self._register(new_table_name, df)
        self.active_table = new_table_name

        # Remember this pair so suggest_joins() stops re-suggesting it now
        # that it's been combined — see suggest_joins() docstring.
        self.combined_pairs.add(frozenset({table_a, table_b}))

        return {
            "tables":         list(self.tables.keys()),
            "active_table":   self.active_table,
            "columns":        self.tables[self.active_table]["columns"],
            "table_details":  self.tables,
            "combined_from":  [table_a, table_b],
            "join_columns":   on_columns,
        }

    def remove_table(self, table_name: str) -> dict:
        """
        Unloads a single table from the session — drops it from DuckDB,
        removes it from self.tables/self.dataframes, and clears any
        combined_pairs entry that references it (a stale "already combined"
        record for a table that no longer exists would otherwise silently
        block a legitimate future suggestion if a table by the same name
        gets loaded again later).

        Works identically regardless of source (uploaded sheet, MySQL
        table, or a prior combine result) — same as every other
        table-scoped method here, removal has no notion of origin.

        If the removed table was the active_table, falls back to another
        currently-loaded table (arbitrarily, the first one left) or None
        if nothing remains.
        """
        if table_name not in self.tables:
            return {"error": f"Table '{table_name}' is not loaded."}

        try:
            self.conn.unregister(table_name)
        except Exception:
            pass  # already gone from the connection, or never registered under this exact name

        del self.tables[table_name]
        self.dataframes.pop(table_name, None)

        self.combined_pairs = {
            pair for pair in self.combined_pairs if table_name not in pair
        }

        if self.active_table == table_name:
            remaining = list(self.tables.keys())
            self.active_table = remaining[0] if remaining else None

        return {
            "tables":        list(self.tables.keys()),
            "active_table":  self.active_table,
            "columns":       self.tables[self.active_table]["columns"] if self.active_table else [],
            "table_details": self.tables,
        }

    def get_joined_dataframe(self, table_a: str, table_b: str, join_type: str = "inner", on_columns: list = None) -> dict:
        """
        Executes a live JOIN between two already-loaded tables and returns
        the resulting DataFrame directly — WITHOUT registering it as a new
        persistent table, unlike combine_tables(). This is what lets
        forecast/trend/anomaly/regression/clustering/classification see
        data that spans two tables: those engines only ever operate on a
        single DataFrame handed to them by main.py's _get_clean_dataframe(),
        so this builds that DataFrame once, on the fly, for one query —
        nothing is kept around afterward, unlike a manual Combine.

        `on_columns`: if given, these EXACT columns are used as the join
        key (validated to exist in both tables) — this is how an explicit
        user join-choice (the join picker) overrides auto-detection. If
        omitted, join columns are auto-detected via _shared_columns() — the
        same composite-key logic suggest_joins() uses for its UI
        suggestions — so automatic behavior never disagrees with what the
        "Combine tables" cards would have proposed.

        `join_type` is "inner" | "left" | "right" | "full", matching
        IntentClassifier's join_type field (and the join picker's choice).

        Only ever joins exactly two tables per call, same scope limit as
        combine_tables() — a question needing three tables' worth of live
        join isn't supported here; a prior manual Combine is the way to
        get that into a single table first.

        Returns {"data": df, "error": None} or {"data": None, "error": "..."},
        matching execute_sql()'s convention.
        """
        missing = [t for t in (table_a, table_b) if t not in self.tables]
        if missing:
            return {"data": None, "error": f"Unknown table(s): {missing}"}

        if on_columns:
            cols_a = self.tables[table_a]["columns"]
            cols_b = self.tables[table_b]["columns"]
            bad_cols = [c for c in on_columns if c not in cols_a or c not in cols_b]
            if bad_cols:
                return {
                    "data": None,
                    "error": f"Column(s) not found in both '{table_a}' and '{table_b}': {bad_cols}",
                }
        else:
            on_columns = self._shared_columns(table_a, table_b)
            if not on_columns:
                return {
                    "data": None,
                    "error": f"No shared columns found between '{table_a}' and '{table_b}' to join on.",
                }

        if join_type not in _JOIN_TYPE_SQL:
            join_type = "inner"

        sql = self._build_join_sql(table_a, table_b, on_columns, join_keyword=_JOIN_TYPE_SQL[join_type])
        return self.execute_sql(sql)

    def get_dataframe(self, table_name: str = None) -> pd.DataFrame:
        name = table_name or self.active_table
        return self.dataframes.get(name)

    def get_all_schemas(self) -> str:
        text = ""
        for table, info in self.tables.items():
            text += f"\nTable: {table}\nColumns:\n"
            for col in info["columns"]:
                dtype = info["dtypes"].get(col, "")
                role  = info.get("roles", {}).get(col)
                note  = " — identifier/code column: NEVER use SUM or AVG on this; use COUNT(*) or COUNT(DISTINCT ...) if a count is needed" if role == "identifier" else ""
                text += f"  - {col} ({dtype}){note}\n"
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
                role  = info.get("roles", {}).get(col)
                note  = " — identifier/code column: NEVER use SUM or AVG on this; use COUNT(*) or COUNT(DISTINCT ...) if a count is needed" if role == "identifier" else ""
                schema_text += f"  - {col} ({dtype}){note}\n"

                # Map column to tables for relationship detection
                if col not in column_map:
                    column_map[col] = []
                column_map[col].append(table)

        # Detect possible joins (columns that exist in >1 table). Uses the
        # same ignore list as suggest_joins() so the AI's join hints and the
        # user-facing auto-join suggestions never disagree on what counts as
        # a meaningful shared column.
        possible_joins = {
            col: tbls for col, tbls in column_map.items()
            if len(tbls) > 1 and col.lower() not in _IGNORE_JOIN_COLS
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