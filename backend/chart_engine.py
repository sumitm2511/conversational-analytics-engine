import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

class ChartEngine:

    def _detect_chart_type(self, df: pd.DataFrame, question: str, chart_type: str | None) -> str:
        if chart_type and chart_type != "auto":
            return chart_type
        q = question.lower()
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols  = df.select_dtypes(include="object").columns.tolist()
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "month", "year", "time", "period"])]

        if len(df) == 1 and len(num_cols) == 1: return "kpi"
        if date_cols and num_cols: return "line"
        if any(w in q for w in ["trend", "over time", "monthly", "weekly", "daily", "growth"]): return "line"
        if any(w in q for w in ["distribution", "histogram", "spread"]): return "histogram"
        if any(w in q for w in ["correlation", "vs", "versus", "compare", "scatter"]):
            if len(num_cols) >= 2: return "scatter"
        if any(w in q for w in ["percentage", "share", "proportion", "breakdown", "composition"]): return "pie"
        if cat_cols and num_cols: return "bar"
        if len(num_cols) >= 2: return "scatter"
        return "bar"

    def _get_colors(self):
        # A beautiful, distinct color palette for multiple lines
        return ["#6366F1", "#10B981", "#F43F5E", "#F59E0B", "#06B6D4", "#8B5CF6", "#3B82F6", "#EC4899"]

    def _base_layout(self, extra: dict = None) -> dict:
        layout = dict(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="#FAFBFC",
            font=dict(family="Inter", size=11, color="#64748B"),
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E2E8F0", borderwidth=1),
            colorway=self._get_colors(),
        )
        if extra: layout.update(extra)
        return layout

    def auto_chart(self, df: pd.DataFrame, question: str, chart_type: str | None) -> dict | None:
        if df is None or df.empty: return None
        df = df.copy().replace([float("inf"), float("-inf")], None)
        ctype     = self._detect_chart_type(df, question, chart_type)
        num_cols  = df.select_dtypes(include="number").columns.tolist()
        cat_cols  = df.select_dtypes(include="object").columns.tolist()
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "month", "year", "time", "period"])]
        colors    = self._get_colors()

        try:
            fig = self._build_chart(df, ctype, num_cols, cat_cols, date_cols, colors, question)
            if fig is None: return None
            fig.update_layout(**self._base_layout())
            result = json.loads(fig.to_json())
            result["suggested_type"] = ctype
            return result
        except Exception as e:
            print(f"Chart error: {e}")
            return None

    def _build_chart(self, df, ctype, num_cols, cat_cols, date_cols, colors, question):
        if ctype == "kpi":
            fig = go.Figure(go.Indicator(
                mode="number", value=df[num_cols[0]].iloc[0],
                title={"text": num_cols[0].replace("_", " ").title(), "font": {"size": 14, "color": "#64748B"}},
                number={"font": {"size": 48, "color": "#0F172A"}}
            ))
            fig.update_layout(height=200)
            return fig

        if ctype == "line":
            x_col   = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else df.columns[0])
            y_col   = num_cols[0]
            color   = cat_cols[1] if len(cat_cols) > 1 else None
            plot_df = df.head(200)
            fig     = go.Figure()

            if color and color in df.columns:
                for i, grp in enumerate(plot_df[color].unique()[:8]):
                    sub = plot_df[plot_df[color] == grp]
                    fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], name=str(grp), mode="lines+markers", line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=5)))
            else:
                fig.add_trace(go.Scatter(x=plot_df[x_col], y=plot_df[y_col], mode="lines+markers", line=dict(color=colors[0], width=2), marker=dict(size=5), fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            fig.update_layout(xaxis_title=x_col.replace("_", " ").title(), yaxis_title=y_col.replace("_", " ").title())
            return fig

        if ctype == "bar":
            x_col   = cat_cols[0] if cat_cols else df.columns[0]
            y_col   = num_cols[0]
            color   = cat_cols[1] if len(cat_cols) > 1 else None
            plot_df = df.head(30)
            if color and color in df.columns:
                fig = px.bar(plot_df, x=x_col, y=y_col, color=color, barmode="group", color_discrete_sequence=colors)
            else:
                fig = px.bar(plot_df, x=x_col, y=y_col, color_discrete_sequence=colors)
            fig.update_traces(marker_line_width=0)
            return fig

        if ctype == "area":
            x_col = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else df.columns[0])
            y_col = num_cols[0]
            return px.area(df.head(200), x=x_col, y=y_col, color_discrete_sequence=colors)

        if ctype == "pie":
            name_col  = cat_cols[0] if cat_cols else df.columns[0]
            value_col = num_cols[0]
            plot_df   = df.head(10)
            return go.Figure(go.Pie(labels=plot_df[name_col], values=plot_df[value_col], hole=0.4, marker_colors=colors, textinfo="label+percent", hovertemplate="%{label}<br>%{value}<br>%{percent}<extra></extra>"))

        if ctype == "scatter":
            x_col = num_cols[0]
            y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
            color = cat_cols[0] if cat_cols else None
            fig   = px.scatter(df.head(500), x=x_col, y=y_col, color=color, color_discrete_sequence=colors, opacity=0.7)
            fig.update_traces(marker=dict(size=7))
            return fig

        if ctype == "histogram":
            y_col = num_cols[0]
            fig   = px.histogram(df, x=y_col, color_discrete_sequence=colors, nbins=30)
            fig.update_traces(marker_line_width=0)
            return fig

        return None

    def prediction_chart(self, df: pd.DataFrame, intent: dict, chart_type: str | None) -> dict | None:
        try:
            ptype  = intent.get("type")
            colors = self._get_colors()

            if df is None or df.empty: return None
            df = df.copy().replace([float("inf"), float("-inf")], None)

            if ptype == "forecast":
                fig  = go.Figure()
                
                date_col  = intent.get("date_col", "date")
                val_col   = intent.get("value_col", "predicted")
                group_col = intent.get("group_col")
                y_col     = "predicted" if "predicted" in df.columns else val_col

                # Internal helper to draw a single country's forecast perfectly
                def _plot_forecast(sub_df, color_idx, name_suffix=""):
                    hist = sub_df[~sub_df["is_forecast"]].copy()
                    fore = sub_df[sub_df["is_forecast"]].copy()
                    
                    if not hist.empty and not fore.empty:
                        last_hist = hist.iloc[[-1]].copy()
                        last_hist["predicted"] = last_hist["actual"]
                        fore = pd.concat([last_hist, fore], ignore_index=True)

                    x_hist = date_col if date_col in hist.columns else hist.columns[0]
                    x_fore = date_col if date_col in fore.columns else fore.columns[0]
                    c = colors[color_idx % len(colors)]
                    
                    hist_name = f"{name_suffix} (History)" if name_suffix else "Historical Truth"
                    fore_name = f"{name_suffix} (Forecast)" if name_suffix else "Predicted Trend"
                    
                    # 1. Plot Historical (Solid Line)
                    fig.add_trace(go.Scatter(x=hist[x_hist], y=hist["actual"], name=hist_name, mode="lines+markers", line=dict(color=c, width=2.5), marker=dict(size=6)))
                    
                    # 2. Plot Confidence Interval (ONLY if we are NOT grouping, to avoid visual clutter)
                    if not group_col and "upper_bound" in fore.columns and "lower_bound" in fore.columns:
                        fig.add_trace(go.Scatter(
                            x=pd.concat([fore[x_fore], fore[x_fore][::-1]]), 
                            y=pd.concat([fore["upper_bound"], fore["lower_bound"][::-1]]), 
                            fill="toself", fillcolor="rgba(245,158,11,0.15)", 
                            line=dict(color="rgba(255,255,255,0)"), 
                            name="80% Confidence Band", showlegend=True, hoverinfo="skip"
                        ))

                    # 3. Plot Forecast (Dotted Line)
                    # If multiple groups, use their unique color. If single, use Orange.
                    fore_color = c if group_col else "#F59E0B"
                    fig.add_trace(go.Scatter(x=fore[x_fore], y=fore[y_col], name=fore_name, mode="lines+markers", line=dict(color=fore_color, width=2.5, dash="dot"), marker=dict(size=6, symbol="diamond")))

                # 🔥 THE FIX: If there are multiple groups, loop through them and draw them all!
                if group_col and group_col in df.columns:
                    for i, grp in enumerate(df[group_col].unique()):
                        _plot_forecast(df[df[group_col] == grp], i, name_suffix=str(grp))
                else:
                    _plot_forecast(df, 0)

                fig.update_layout(title="Forecast Projection", xaxis_title="Date", yaxis_title="Value", legend=dict(orientation="h", y=-0.15))

            elif ptype == "regression":
                val_col = intent.get("value_col") or intent.get("target_col")
                if not val_col or val_col not in df.columns:
                    num_cols = df.select_dtypes(include="number").columns
                    val_col  = num_cols[0] if len(num_cols) > 0 else None
                if not val_col: return None

                fig = px.scatter(df, x=val_col, y="predicted", color_discrete_sequence=colors, opacity=0.6, labels={val_col: "Actual", "predicted": "Predicted"}, title="Actual vs Predicted")
                mn, mx = df[val_col].min(), df[val_col].max()
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect Fit", line=dict(color="#F43F5E", dash="dot", width=1.5)))

            elif ptype == "trend":
                val_col  = intent.get("value_col")
                date_col = intent.get("date_col")
                fig      = go.Figure()

                if date_col and date_col in df.columns and val_col and val_col in df.columns:
                    fig.add_trace(go.Scatter(x=df[date_col], y=df[val_col], name="Actual", line=dict(color="#6366F1", width=2), fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
                    if "rolling_3" in df.columns: fig.add_trace(go.Scatter(x=df[date_col], y=df["rolling_3"], name="3-Period MA", line=dict(color="#F59E0B", width=2, dash="dash")))
                    if "trend_component" in df.columns: fig.add_trace(go.Scatter(x=df[date_col], y=df["trend_component"], name="Trend", line=dict(color="#F43F5E", width=1.5, dash="longdash")))
                fig.update_layout(title="Trend Analysis")

            elif ptype == "anomaly":
                val_col  = intent.get("value_col")
                date_col = intent.get("date_col")
                fig      = go.Figure()

                if val_col and val_col in df.columns:
                    x_vals = df[date_col] if (date_col and date_col in df.columns) else df.index
                    fig.add_trace(go.Scatter(x=x_vals, y=df[val_col], name="Value", mode="lines", line=dict(color="#6366F1", width=1.5)))

                    if "is_anomaly" in df.columns:
                        anom = df[df["is_anomaly"]]
                        x_anom = anom[date_col] if (date_col and date_col in anom.columns) else anom.index
                        fig.add_trace(go.Scatter(x=x_anom, y=anom[val_col], name="Anomaly", mode="markers", marker=dict(color="#F43F5E", size=10, symbol="x")))

                fig.update_layout(title="Anomaly Detection")
            else: return None

            fig.update_layout(**self._base_layout())
            result = json.loads(fig.to_json())
            result["suggested_type"] = ptype
            return result

        except Exception as e:
            print(f"Prediction chart error: {e}")
            return None


_engine = ChartEngine()

def build_auto_chart(df, question, chart_type=None):
    return _engine.auto_chart(df, chart_type, question)

def build_forecast_chart(result, date_col, value_col):
    data   = result["data"]
    
    # 🔥 THE FIX: If the data is a dictionary (multiple groups), stitch them together safely!
    if isinstance(data, dict):
        parts = []
        for grp, df_grp in data.items():
            df_grp = df_grp.copy()
            df_grp["_group_col"] = str(grp)
            parts.append(df_grp)
        data = pd.concat(parts, ignore_index=True)
        intent = {"type": "forecast", "date_col": date_col, "value_col": value_col, "group_col": "_group_col"}
    else:
        intent = {"type": "forecast", "date_col": date_col, "value_col": value_col}
        
    return _engine.prediction_chart(data, intent, None)

def build_trend_chart(df, date_col, value_col, group_col=None):
    intent = {"type": "trend", "date_col": date_col, "value_col": value_col}
    return _engine.prediction_chart(df, intent, None)

def build_regression_chart(result):
    intent = {"type": "regression", "target_col": result.get("target")}
    return _engine.prediction_chart(result["data"], intent, None)

def build_fi_chart(fi_df):
    return _engine.auto_chart(fi_df, "feature importance", "bar")

def build_anomaly_chart(result, date_col, value_col):
    df = result.get("data")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        df = result.get("anomalies")
        
    intent = {"type": "anomaly", "date_col": date_col, "value_col": value_col}
    return _engine.prediction_chart(df, intent, None) if df is not None else None