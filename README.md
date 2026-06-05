# Conversational Analytics Engine

Conversational Analytics Engine is a full-stack analytics application that lets a user upload CSV or Excel data, ask questions in natural language, and receive structured results as tables, KPIs, charts, SQL, and AI-generated business insights.

The current codebase is split into:

```text
Frontend (React + Vite)
\- Collects files and questions from the user

Backend (FastAPI)
\- Classifies intent, runs analytics, generates charts, and returns insights

In-memory data layer (DuckDB + Pandas)
\- Loads uploaded files as session-scoped tables for querying and modeling

LLM layer (Groq)
\- Intent routing, SQL generation/fixing, and narrative insight generation
```

## What This Project Does

- Uploads `.csv`, `.xlsx`, and `.xls` files.
- Converts uploaded data into DuckDB in-memory tables.
- Supports conversational analysis over the uploaded data.
- Routes questions into one of these analysis modes:
  - `sql`
  - `summary`
  - `forecast`
  - `trend`
  - `anomaly`
  - `regression`
- Returns:
  - preview rows
  - total row count
  - KPI metrics
  - Plotly chart JSON
  - generated SQL for SQL-style queries
  - AI-written business summary, findings, anomaly notes, and follow-up questions

## Tech Stack

| Layer | Tools used in repo | Purpose |
| --- | --- | --- |
| Frontend | React 18, Vite, Axios, Plotly, `react-plotly.js`, `uuid`, `lucide-react` | UI, chart rendering, API calls, client-side session id |
| Backend API | FastAPI, Uvicorn, Pydantic, `python-multipart` | HTTP API, file upload, request validation |
| Data processing | Pandas, DuckDB, OpenPyXL, XLRD | File ingestion, schema detection, SQL execution |
| AI / LLM | Groq API | Intent classification, SQL generation/fixing, insight generation |
| Forecasting / ML | Prophet, Statsmodels, PMDARIMA, Scikit-learn, NumPy | Forecasting, trend analysis, anomaly detection, regression |
| Visualization | Plotly, Plotly Express, Plotly Graph Objects | Backend chart generation + frontend rendering |
| Packaging / deployment | Docker, Docker Compose, Nginx | Containerized deployment and reverse proxy |

## Repository Structure

```text
conversational-analytics-engine/
|- backend/
|  |- .env                       # environment file; needs GROQ_API_KEY
|  |- Dockerfile                 # FastAPI container
|  |- requirements.txt           # Python dependencies
|  |- main.py                    # FastAPI app, routes, session store, dispatcher
|  |- analytics_engine.py        # file loading, schema detection, DuckDB execution
|  |- intent_classifier.py       # Groq-based intent classification
|  |- sql_generator.py           # Groq-based SQL generation and SQL repair
|  |- prediction_engine.py       # forecast, trend, regression, anomaly engines
|  |- chart_engine.py            # Plotly chart builders
|  |- insight_engine.py          # Groq-based narrative insight generation
|  |- uvicorn.out.log            # local runtime log artifact
|  \- uvicorn.err.log            # local runtime log artifact
|- frontend/
|  |- .gitignore
|  |- Dockerfile                 # Vite build -> Nginx static serving
|  |- index.html                 # Vite entry html
|  |- package.json               # frontend scripts and dependencies
|  |- package-lock.json
|  \- src/
|     |- main.jsx                # React entrypoint
|     |- App.jsx                 # top-level app layout and session wiring
|     |- App.css                 # full application styling
|     |- config.js               # API base url
|     \- components/
|        |- TopBar.jsx           # header / chrome
|        |- FileUpload.jsx       # file upload UI
|        |- ChatPanel.jsx        # prompt bar and suggestion chips
|        |- ResultPanel.jsx      # tabs, KPIs, chart/table/sql/insight display
|        |- ChartPanel.jsx       # Plotly chart renderer and chart type switching
|        |- SchemaPanel.jsx      # table and column browser
|        |- KPIBar.jsx           # dataset-level KPIs
|        |- ExportMenu.jsx       # export CSV/JSON/XLS/chart
|        |- SkeletonCard.jsx     # loading placeholder
|        \- ForecastPanel.jsx    # present in repo, currently not mounted
|- docker-compose.yml            # multi-container orchestration
|- nginx.conf                    # reverse proxy config
\- Book1.xlsx                    # sample workbook included in repo
```

## How The System Runs

### High-level runtime tree

```text
Browser
\- React app
   |- generates a UUID session id on load
   |- uploads a file to FastAPI
   |- sends natural-language questions to FastAPI
   \- renders charts/tables/insights returned by the backend

FastAPI app
\- per-session in-memory state
   |- AnalyticsEngine
   |  |- loads file into Pandas
   |  |- normalizes column names
   |  |- registers tables in DuckDB
   |  \- exposes schema + SQL execution
   |- AnalyticsDispatcher
   |  |- asks LLM to classify the question
   |  |- chooses the right analysis path
   |  |- runs SQL or ML/statistical engine
   |  |- generates a chart
   |  \- generates business insights
   \- returns JSON to frontend
```

### Upload flow tree

```text
User selects file
\- Frontend: FileUpload.jsx
   \- POST /upload/{session_id}
      \- FastAPI: upload_file()
         |- validates extension (.csv, .xlsx, .xls)
         |- writes upload to a temporary file
         |- AnalyticsEngine.load_file()
         |  |- CSV -> one table named sheet1
         |  |- Excel -> one table per sheet
         |  |- sanitizes column names
         |  |- infers basic dtypes
         |  |- stores raw DataFrames
         |  \- registers tables in DuckDB memory
         |- InsightEngine.summarize_dataset()
         |- deletes temporary file
         \- returns schema + active table + overview
```

### Query flow tree

```text
User asks a question
\- Frontend: ChatPanel.jsx
   \- POST /query/{session_id}
      \- FastAPI: query()
         \- AnalyticsDispatcher.handle_query()
            |- IntentClassifier.classify()
            |  \- Groq returns one intent:
            |     |- sql
            |     |- summary
            |     |- forecast
            |     |- trend
            |     |- anomaly
            |     \- regression
            |- sql / summary
            |  |- SQLGenerator.generate_sql()
            |  |- DuckDB executes query
            |  |- SQLGenerator.fix_sql() on failure
            |  |- ChartEngine builds auto chart
            |  \- InsightEngine writes findings
            |- forecast
            |  |- ForecastEngine chooses best model
            |  |- models include Prophet / ETS / ARIMA / linear
            |  |- ChartEngine builds forecast chart
            |  \- InsightEngine writes forecast narrative
            |- trend
            |  |- TrendEngine computes rolling stats and trend features
            |  |- ChartEngine builds trend chart
            |  \- InsightEngine writes trend narrative
            |- anomaly
            |  |- AnomalyEngine uses rolling z-score or isolation forest
            |  |- ChartEngine builds anomaly chart
            |  \- InsightEngine writes anomaly narrative
            \- regression
               |- RegressionEngine trains GradientBoostingRegressor
               |- feature importance is computed
               |- ChartEngine builds regression / FI chart
               \- InsightEngine writes driver analysis
```

## Backend Module Responsibilities

| File | Responsibility |
| --- | --- |
| `backend/main.py` | FastAPI bootstrapping, CORS, request models, per-session state, upload/query/history/session routes, analytics dispatch |
| `backend/analytics_engine.py` | Reads CSV/Excel, sanitizes column names, infers schema, registers DuckDB tables, runs SQL |
| `backend/intent_classifier.py` | Sends schema + user question to Groq and returns structured routing metadata |
| `backend/sql_generator.py` | Generates DuckDB SQL from natural language and retries with repair on SQL errors |
| `backend/prediction_engine.py` | Contains forecasting, trend, anomaly, and regression logic |
| `backend/chart_engine.py` | Converts result DataFrames into Plotly JSON payloads |
| `backend/insight_engine.py` | Produces executive summaries, bullets, anomaly flags, and next-question suggestions |

## Frontend Module Responsibilities

| File | Responsibility |
| --- | --- |
| `frontend/src/App.jsx` | App shell, generated session id, overall state, sidebar behavior, result feed |
| `frontend/src/config.js` | Chooses API base URL from `VITE_API_URL` or defaults to `http://127.0.0.1:8000` |
| `frontend/src/components/FileUpload.jsx` | Upload control and file POST request |
| `frontend/src/components/ChatPanel.jsx` | Prompt entry, chart type selector, suggestion chips |
| `frontend/src/components/ResultPanel.jsx` | Renders summary/chart/table/sql/insights tabs and follow-up actions |
| `frontend/src/components/ChartPanel.jsx` | Plotly rendering, chart switcher, fullscreen mode |
| `frontend/src/components/SchemaPanel.jsx` | Shows uploaded tables and columns in the sidebar |
| `frontend/src/components/KPIBar.jsx` | Dataset summary cards for rows, columns, tables, data health |
| `frontend/src/components/ExportMenu.jsx` | Exports table data and chart images |
| `frontend/src/components/TopBar.jsx` | UI chrome; current export/settings buttons are visual only |
| `frontend/src/components/ForecastPanel.jsx` | Present but not used by `App.jsx` / `ResultPanel.jsx` right now |

## API Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/upload/{session_id}` | Upload CSV/Excel file and initialize the session dataset |
| `POST` | `/query/{session_id}` | Ask a natural-language analytics question |
| `GET` | `/history/{session_id}` | Return recent question history for the session |
| `DELETE` | `/session/{session_id}` | Clear session memory |
| `GET` | `/health` | Health check |

### Response behavior worth knowing

- Result rows are capped to the first `500` rows in API responses.
- Query history is capped to the most recent `20` entries.
- Sessions are stored in Python memory only.
- Uploaded files are loaded into memory and the temp upload file is deleted after processing.
- There is no persistent database in the current codebase.

## Supported Analysis Paths

```text
Natural-language question
\- Intent classification
   |- SQL / Summary
   |  \- DuckDB query over uploaded table(s)
   |- Forecast
   |  \- time-series projection
   |- Trend
   |  \- rolling averages, momentum, CAGR, spikes
   |- Anomaly
   |  \- outlier detection
   \- Regression
      \- target vs feature modeling + feature importance
```

## Local Development Setup

Local development is the cleanest way to run the project in its current state.

### 1. Prerequisites

- Python `3.11` recommended
- Node.js `20+`
- `npm`
- A valid Groq API key

### 2. Backend setup

From the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create or update `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Start the API:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 3. Frontend setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend will be available at:

```text
http://localhost:5173
```

The default frontend API target already points to:

```text
http://127.0.0.1:8000
```

If you want to override that:

```powershell
$env:VITE_API_URL="http://127.0.0.1:8000"
npm run dev
```

### 4. First run checklist

```text
1. Start backend on port 8000
2. Start frontend on port 5173
3. Open the browser
4. Upload Book1.xlsx or your own CSV/XLSX/XLS file
5. Ask questions such as:
   |- Show total sales by region
   |- Forecast next 6 months
   |- Detect anomalies in revenue
   \- Which factors influence billability
```

## Docker Setup

Docker support exists in the repo, but the checked-in reverse-proxy path is not fully aligned yet. Use local development first if you want the least friction.

### What exists

```text
docker-compose.yml
|- backend service
|- frontend service
\- nginx service

backend/Dockerfile
\- Python 3.11 + FastAPI + Uvicorn

frontend/Dockerfile
\- Node build stage -> Nginx static serving stage
```

### Current caveats in checked-in config

```text
nginx.conf
\- proxies frontend traffic to frontend:3000
   \- but the frontend container serves Nginx on port 80, not 3000

frontend config
\- defaults API calls to http://127.0.0.1:8000
   \- so reverse-proxy style /api routing only works if VITE_API_URL is set at build time
```

### If you still want to use Docker

You will need:

- `GROQ_API_KEY` exported in your shell or `.env` for Compose
- frontend built with `VITE_API_URL=/api`
- `nginx.conf` updated so the frontend upstream matches the container's actual port

Then run:

```powershell
docker compose up --build
```

## Data Handling Rules In The Current Code

- CSV files are loaded as a single table named `sheet1`.
- Excel files are loaded as one table per worksheet.
- Column names are lowercased, stripped, and converted to underscore-separated identifiers.
- Duplicate column names are made unique.
- Datelike object columns are heuristically converted to datetimes.
- Join hints are inferred when the same column exists across multiple tables.

## Session Model

```text
Page load
\- frontend creates one UUID session id
   \- every upload and query uses that same id
      \- backend stores:
         |- AnalyticsEngine instance
         \- last 20 query history entries
```

This means:

- one browser page load maps to one in-memory analytics session
- restarting the backend clears all sessions
- session data is not persisted to disk or an external database

## Charts And Output Types

The UI can request or render these chart types:

- `auto`
- `bar`
- `line`
- `area`
- `pie`
- `scatter`
- `histogram`

Returned result cards may include:

- summary tab
- chart tab
- table tab
- SQL tab
- insights tab
- export options for table/chart assets

## Recommended Questions To Try

- `Show total sales by region`
- `Top 10 employees by utilization`
- `Forecast next 3 months`
- `Show trend over time`
- `Detect anomalies in revenue`
- `Which factors influence billability`
- `Average utilization by domain`

## Operational Notes

- CORS is currently open for `localhost:5173`, `127.0.0.1:5173`, `localhost:3000`, and `127.0.0.1:3000`.
- The frontend sidebar can collapse/pin and shows schema metadata after upload.
- The backend uses shared singleton helper objects for LLM and analytics services, while each session gets its own `AnalyticsEngine`.
- The result UI supports exporting table data as CSV, JSON, and Excel-compatible TSV, plus chart export as PNG or SVG.

## Known Gaps / Implementation Notes

- `frontend/src/components/ForecastPanel.jsx` exists but is not currently rendered.
- `TopBar.jsx` includes Export and Settings buttons without connected behavior.
- Docker Compose + `nginx.conf` need small alignment changes before the reverse-proxy path works cleanly out of the box.
- Because Groq is used for routing, SQL generation, and insights, the app depends on external LLM availability even for normal query flows.

## Minimal Start Command Summary

```text
Terminal 1
\- cd backend
   |- python -m venv .venv
   |- .venv\Scripts\Activate.ps1
   |- pip install -r requirements.txt
   \- uvicorn main:app --reload --host 0.0.0.0 --port 8000

Terminal 2
\- cd frontend
   |- npm install
   \- npm run dev
```

Then open:

```text
http://localhost:5173
```
