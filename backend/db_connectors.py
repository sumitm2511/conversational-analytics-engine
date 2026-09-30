"""
db_connectors.py — MySQL and Snowflake connectors for AnalyticsEngine.

Exposes:
  - MySQLConnector.run_query(sql)        -> pandas DataFrame
  - MySQLConnector.list_tables()         -> [{"name": ..., "row_count": ...}, ...]
  - SnowflakeConnector.run_query(sql)    -> pandas DataFrame
  - SnowflakeConnector.list_tables()     -> [{"name": ..., "row_count": ...}, ...]

Both list_tables() implementations read from each database's own metadata
views (information_schema.TABLES / Snowflake's INFORMATION_SCHEMA.TABLES)
rather than running SELECT COUNT(*) per table — a metadata lookup instead
of a full scan, so listing tables stays fast even on large databases. Row
counts from metadata are estimates in both engines (not a live exact
count) — fine for a sidebar display.

Loaded data funnels into AnalyticsEngine._register_dataframe() exactly like
an uploaded CSV/Excel sheet — same schema detection, health scoring,
DuckDB registration downstream, regardless of which connector produced it.

Required packages (add to requirements.txt):
  mysql-connector-python
  sqlalchemy
  snowflake-connector-python[pandas]
  cryptography

── SNOWFLAKE AUTH ───────────────────────────────────────────────────────────
Key-pair (JWT) auth ONLY — confirmed via Ericsson's IDAP platform docs that
username/password and interactive SSO are explicitly DISABLED for all
IDAP-provisioned Snowflake service accounts, so there is no password
fallback path in this connector. See SnowflakeConnector's own docstring
below for the full provisioning flow (Functional ID via iSignum, IDAP-
generated key pair, IDAP-created warehouse, mandatory network policy for
this backend's egress IP) and the exact required env vars — none of this
is self-service, all of it is provisioned by IDAP as part of an onboarding
Work Order.

SNOWFLAKE_ACCOUNT uses the NEW org-name format ("<orgname>-<accountname>"),
which IDAP is standardizing on org-wide — get the exact string for DEV/
TEST/PROD from IDAP for your target environment.
"""

import os

import pandas as pd


class MySQLConnector:
    def __init__(self):
        self.host     = os.getenv("MYSQL_HOST")
        self.port     = os.getenv("MYSQL_PORT", "3306")
        self.user     = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DATABASE")

        missing = [name for name, val in [
            ("MYSQL_HOST", self.host),
            ("MYSQL_USER", self.user),
            ("MYSQL_PASSWORD", self.password),
            ("MYSQL_DATABASE", self.database),
        ] if not val]
        if missing:
            raise RuntimeError(
                f"Missing required MySQL env vars: {', '.join(missing)}. "
                "Set these in your .env file."
            )

    def _engine(self):
        from sqlalchemy import create_engine
        # mysql+mysqlconnector — matches the connection pattern already
        # validated by the DB team for this database, via mysql-connector-python.
        url = (
            f"mysql+mysqlconnector://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        return create_engine(url)

    def run_query(self, sql: str) -> pd.DataFrame:
        """Runs a read query and returns the result as a DataFrame."""
        engine = self._engine()
        try:
            return pd.read_sql(sql, engine)
        finally:
            engine.dispose()

    def list_tables(self) -> list:
        """
        Returns [{"name": <table>, "row_count": <int or None>}, ...] for
        every table in the configured database. row_count comes from
        information_schema's TABLE_ROWS, which is an estimate for InnoDB
        (not an exact live count) — fine for a sidebar display, and much
        cheaper than SELECT COUNT(*) per table.
        """
        engine = self._engine()
        try:
            query = (
                "SELECT TABLE_NAME AS name, TABLE_ROWS AS row_count "
                "FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = %(db)s "
                "ORDER BY TABLE_NAME"
            )
            df = pd.read_sql(query, engine, params={"db": self.database})
            df["row_count"] = df["row_count"].fillna(0).astype(int)
            return df.to_dict("records")
        finally:
            engine.dispose()


class SnowflakeConnector:
    """
    Key-pair (JWT) auth ONLY — confirmed via Ericsson's IDAP platform docs
    that username/password and interactive SSO are explicitly DISABLED for
    all IDAP-provisioned Snowflake service accounts. There is no password
    fallback here on purpose: a fallback that can never actually work on
    this org's accounts is worse than no fallback, since it would silently
    invite a misconfigured .env to "look" supported and fail confusingly
    later instead of failing loudly and correctly at startup.

    Provisioning is NOT self-service — per IDAP docs this requires:
      1. Your team creates a Functional ID (ID Type = "Application Id") via
         iSignum, and raises a Work Order in My Support & Services.
      2. IDAP SCIMs the Functional ID into Snowflake as a service user (no
         interactive login/MFA path exists for it).
      3. IDAP generates the RSA key pair (2048-bit minimum, 4096-bit
         recommended) — passphrase is MANDATORY, private key delivered in
         PKCS#8/AES-256 format.
      4. IDAP also creates the warehouse (WH_<applicationname>_<Environment>,
         Gen2, auto-suspend enforced, resource monitor attached) and a
         dedicated network policy allowlisting this backend's egress IP —
         get that IP from your deployment platform (e.g. your Glean/hosting
         contact) and pass it to IDAP as part of the same Work Order, since
         "Request blocked / IP not allowed" is the failure mode if it's
         missing.

    Required env vars (all mandatory — matches the exact config block IDAP
    hands back on provisioning):
      SNOWFLAKE_ACCOUNT              e.g. "<orgname>-<accountname>" — the
                                     NEW org-name format IDAP is standardizing
                                     on; get the exact string for your target
                                     environment (DEV/TEST/PROD map to
                                     develop/test/main branches) from IDAP.
      SNOWFLAKE_USER                 the Functional/Application ID, e.g. "APP_FUNCTIONAL_ID"
      SNOWFLAKE_ROLE                 the MDA role IDAP assigns, e.g. "MDA_ROLE"
      SNOWFLAKE_WAREHOUSE            "WH_<applicationname>_<Environment>" — IDAP-created, not self-service
      SNOWFLAKE_DATABASE
      SNOWFLAKE_SCHEMA
      SNOWFLAKE_PRIVATE_KEY_PATH     path to the .p8 (PKCS#8) key file IDAP delivers
      SNOWFLAKE_PRIVATE_KEY_PASSPHRASE   mandatory — IDAP-generated keys are always passphrase-protected
    """

    def __init__(self):
        self.account   = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user      = os.getenv("SNOWFLAKE_USER")
        self.role      = os.getenv("SNOWFLAKE_ROLE")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database  = os.getenv("SNOWFLAKE_DATABASE")
        self.schema    = os.getenv("SNOWFLAKE_SCHEMA")

        self.private_key_path       = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")

        missing = [name for name, val in [
            ("SNOWFLAKE_ACCOUNT", self.account),
            ("SNOWFLAKE_USER", self.user),
            ("SNOWFLAKE_ROLE", self.role),
            ("SNOWFLAKE_WAREHOUSE", self.warehouse),
            ("SNOWFLAKE_DATABASE", self.database),
            ("SNOWFLAKE_SCHEMA", self.schema),
            ("SNOWFLAKE_PRIVATE_KEY_PATH", self.private_key_path),
            ("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", self.private_key_passphrase),
        ] if not val]
        if missing:
            raise RuntimeError(
                f"Missing required Snowflake env vars: {', '.join(missing)}. "
                "Set these in your .env file. Note: password/SSO auth is "
                "disabled on IDAP Snowflake accounts — key-pair (all of the "
                "above) is the only supported path, and the account, user, "
                "role, warehouse, and key file are all provisioned by IDAP, "
                "not chosen freely — see this class's docstring."
            )

    def _load_private_key_bytes(self):
        """
        Loads and decrypts the PEM/PKCS#8 private key file IDAP delivers,
        returning the DER-encoded bytes snowflake-connector-python's
        `private_key` connect() param expects. The passphrase is always
        present here (enforced as mandatory above) since IDAP-generated
        keys are never delivered unencrypted.
        """
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        with open(self.private_key_path, "rb") as f:
            key_bytes = f.read()

        private_key = serialization.load_pem_private_key(
            key_bytes,
            password=self.private_key_passphrase.encode(),
            backend=default_backend(),
        )
        return private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def _connect(self):
        import snowflake.connector

        return snowflake.connector.connect(
            account=self.account,
            user=self.user,
            role=self.role,
            authenticator="SNOWFLAKE_JWT",
            private_key=self._load_private_key_bytes(),
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
        )

    def run_query(self, sql: str) -> pd.DataFrame:
        """
        Runs a read query and returns the result as a DataFrame, via
        fetch_pandas_all() (requires the `snowflake-connector-python[pandas]`
        extra — pulls in pyarrow) rather than manually building a DataFrame
        from cursor.fetchall(), since fetch_pandas_all() preserves Snowflake's
        own column typing (numeric/date/etc.) instead of everything coming
        back as generic Python objects.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(sql)
                return cur.fetch_pandas_all()
            finally:
                cur.close()
        finally:
            conn.close()

    def list_tables(self) -> list:
        """
        Returns [{"name": <table>, "row_count": <int or None>}, ...] for
        every table in the configured database/schema. ROW_COUNT comes from
        Snowflake's own INFORMATION_SCHEMA.TABLES metadata (refreshed
        periodically by Snowflake, not a live exact count) — same
        metadata-over-full-scan tradeoff as MySQLConnector.list_tables().
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT TABLE_NAME AS NAME, ROW_COUNT "
                    "FROM INFORMATION_SCHEMA.TABLES "
                    "WHERE TABLE_SCHEMA = %(schema)s AND TABLE_TYPE = 'BASE TABLE' "
                    "ORDER BY TABLE_NAME",
                    {"schema": self.schema},
                )
                df = cur.fetch_pandas_all()
            finally:
                cur.close()
        finally:
            conn.close()

        df.columns = [c.lower() for c in df.columns]
        df["row_count"] = df["row_count"].fillna(0).astype(int)
        return df.to_dict("records")