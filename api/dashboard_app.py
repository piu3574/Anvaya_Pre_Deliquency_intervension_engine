from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from supabase import create_client, Client
import os
import joblib
import pandas as pd
import numpy as np
import pickle
import json
import time
import datetime

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

app = Flask(__name__)
CORS(app)

# Credentials
SUPABASE_URL = os.getenv(
    "SUPABASE_URL", "https://fotkkamptuylqubvwyom.supabase.co"
).strip()
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig",
).strip()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Artifacts for V3 Turbo
MODELS_DIR = os.path.join("modeltraining", "artifacts")
m_xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
m_lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
m_meta = joblib.load(os.path.join(MODELS_DIR, "ensemble_meta.pkl"))
m_scaler = joblib.load(os.path.join(MODELS_DIR, "meta_scaler.pkl"))

with open(os.path.join(MODELS_DIR, "banding_config.pkl"), "rb") as f:
    banding_config = pickle.load(f)

FEATURE_NAMES = [
    "stress_f1",
    "stress_f2",
    "stress_f3",
    "stress_f5",
    "stress_f6",
    "stress_f14",
    "vintage",
    "age",
    "income_vol",
    "overdraft",
]

PIPELINE_STAGES = [
    "Event Trigger",
    "Capture & Ingestion",
    "Feature Engineering",
    "Fast Path Scoring",
    "Deep Path Scoring",
    "Ensemble Scoring",
    "Risk Banding",
    "Persistence",
    "Explainability",
    "Intervention Routing",
]


def _parse_iso(ts: str):
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_risk_log(row: dict):
    customer_id = row.get("external_customer_id") or row.get("customer_id") or "UNKNOWN"
    pd_ensemble = row.get("pd_ensemble")
    if pd_ensemble is None:
        pd_ensemble = row.get("pd_final")
    pd_ensemble = float(pd_ensemble or 0)

    risk_band = (row.get("risk_band") or "YELLOW").upper()
    scored_at = row.get("scored_at") or row.get("created_at")

    return {
        "id": row.get("id"),
        "customer_id": customer_id,
        "pd_xgb": float(row.get("pd_xgb") or 0),
        "pd_lgbm": float(row.get("pd_lgbm") or 0),
        "pd_ensemble": pd_ensemble,
        "risk_band": risk_band,
        "scored_at": scored_at,
        "source": row.get("source") or "unknown",
        "priority": "critical"
        if risk_band == "RED"
        else ("watch" if risk_band == "YELLOW" else "normal"),
    }


def _fetch_recent_risk_logs(limit=300):
    res = (
        supabase.table("risk_scores")
        .select("*")
        .order("scored_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data if res and res.data else []


def _persist_risk_log(
    external_customer_id: str,
    p_xgb: float,
    p_lgbm: float,
    p_ensemble: float,
    risk_band: str,
    t_g: float,
    t_r: float,
):
    try:
        payload = {
            "external_customer_id": external_customer_id,
            "pd_xgb": float(p_xgb),
            "pd_lgbm": float(p_lgbm),
            "pd_ensemble": float(p_ensemble),
            "risk_band": risk_band,
            "t_green": float(t_g),
            "t_red": float(t_r),
            "source": "dashboard_predict_v3",
        }
        supabase.table("risk_scores").insert(payload).execute()
    except Exception:
        return


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "ok", "service": "anvaya-dashboard-api", "version": "3.0.0-turbo"}
    )


@app.route("/stats", methods=["GET"])
def stats():
    try:
        res = (
            supabase.table("dashboard_customers")
            .select("pd_final, risk_band", count="exact")
            .execute()
        )
        count = res.count
        data = res.data
        if not data:
            return jsonify({"count": 0, "msg": "No data in table"})

        df = pd.DataFrame(data)
        return jsonify(
            {
                "total_count_in_db": count,
                "sample_count": len(df),
                "pd_final_stats": {
                    "min": float(df["pd_final"].min()),
                    "mean": float(df["pd_final"].mean()),
                    "max": float(df["pd_final"].max()),
                },
                "distribution_sample": df["risk_band"].value_counts().to_dict(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs/risk", methods=["GET"])
def risk_logs():
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        band = request.args.get("risk_band")

        query = (
            supabase.table("risk_scores")
            .select("*", count="exact")
            .order("scored_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if band:
            query = query.eq("risk_band", band.upper())

        res = query.execute()
        rows = res.data if res and res.data else []
        normalized = [_normalize_risk_log(r) for r in rows]
        return jsonify({"count": res.count or len(normalized), "items": normalized})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs/risk/stream", methods=["GET"])
def risk_logs_stream():
    poll_seconds = float(request.args.get("poll_seconds", 3.0))
    limit = int(request.args.get("limit", 30))

    def event_stream():
        last_seen_key = None
        while True:
            try:
                rows = _fetch_recent_risk_logs(limit=limit)
                normalized = [_normalize_risk_log(r) for r in rows]
                normalized_sorted = sorted(
                    normalized,
                    key=lambda r: (r.get("scored_at") or "", str(r.get("id") or "")),
                )
                for item in normalized_sorted:
                    event_key = f"{item.get('scored_at')}|{item.get('id')}"
                    if last_seen_key is None or event_key > last_seen_key:
                        yield f"data: {json.dumps(item)}\n\n"
                        last_seen_key = event_key

            except Exception as stream_error:
                yield f"event: error\ndata: {json.dumps({'error': str(stream_error)})}\n\n"

            time.sleep(max(1.0, poll_seconds))

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/pipeline/status", methods=["GET"])
def pipeline_status():
    try:
        health_data = health().get_json()
        logs = _fetch_recent_risk_logs(limit=120)
        normalized = [_normalize_risk_log(r) for r in logs]

        last_event = normalized[0] if normalized else None
        last_time = _parse_iso(last_event.get("scored_at")) if last_event else None
        now = datetime.datetime.now(datetime.timezone.utc)

        seconds_since_last = None
        if last_time:
            seconds_since_last = max(0, int((now - last_time).total_seconds()))

        if seconds_since_last is None:
            pipeline_state = "cold"
        elif seconds_since_last <= 120:
            pipeline_state = "active"
        elif seconds_since_last <= 900:
            pipeline_state = "idle"
        else:
            pipeline_state = "stale"

        recent_df = pd.DataFrame(normalized)
        if recent_df.empty:
            events_15m = 0
            red_15m = 0
        else:
            recent_df["scored_at"] = pd.to_datetime(
                recent_df["scored_at"], errors="coerce", utc=True
            )
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
            in_window = recent_df[recent_df["scored_at"] >= cutoff]
            events_15m = int(len(in_window))
            red_15m = (
                int((in_window["risk_band"] == "RED").sum())
                if not in_window.empty
                else 0
            )

        return jsonify(
            {
                "state": pipeline_state,
                "service": health_data,
                "last_scored_at": last_event.get("scored_at") if last_event else None,
                "seconds_since_last_score": seconds_since_last,
                "events_last_15m": events_15m,
                "red_events_last_15m": red_15m,
                "latest_event": last_event,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pipeline/runs", methods=["GET"])
def pipeline_runs():
    try:
        horizon_hours = int(request.args.get("hours", 24))
        rows = _fetch_recent_risk_logs(limit=1000)
        normalized = [_normalize_risk_log(r) for r in rows]
        df = pd.DataFrame(normalized)
        if df.empty:
            return jsonify({"items": []})

        df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=horizon_hours)
        df = df[df["scored_at"] >= cutoff]
        if df.empty:
            return jsonify({"items": []})

        df["hour"] = df["scored_at"].dt.floor("h")
        grouped = (
            df.groupby("hour")
            .agg(
                total_events=("customer_id", "count"),
                critical_events=("risk_band", lambda x: int((x == "RED").sum())),
                watch_events=("risk_band", lambda x: int((x == "YELLOW").sum())),
            )
            .reset_index()
            .sort_values("hour", ascending=False)
        )

        items = []
        for _, row in grouped.iterrows():
            items.append(
                {
                    "run_window": row["hour"].isoformat(),
                    "total_events": int(row["total_events"]),
                    "critical_events": int(row["critical_events"]),
                    "watch_events": int(row["watch_events"]),
                    "status": "critical" if row["critical_events"] > 0 else "healthy",
                }
            )
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pipeline/stages/latest", methods=["GET"])
def pipeline_stages_latest():
    try:
        status_res = pipeline_status().get_json()
        latest = status_res.get("latest_event") or {}
        state = status_res.get("state", "unknown")

        stages = []
        for i, name in enumerate(PIPELINE_STAGES, start=1):
            stage_state = "ok"
            detail = "system check passed"
            if state in ["stale", "cold"] and i >= 4:
                stage_state = "blocked"
                detail = "no fresh scoring events"
            if i == 8 and not latest:
                stage_state = "blocked"
                detail = "no persisted score logs"
            stages.append(
                {"stage": i, "name": name, "status": stage_state, "detail": detail}
            )

        return jsonify({"items": stages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-health/summary", methods=["GET"])
def model_health_summary():
    try:
        res = (
            supabase.table("dashboard_customers")
            .select("pd_final, risk_band, y_true")
            .execute()
        )
        data = res.data if res and res.data else []
        if not data:
            return jsonify(
                {"message": "No model-health data available", "has_data": False}
            )

        df = pd.DataFrame(data)
        df["pd_final"] = pd.to_numeric(df["pd_final"], errors="coerce")
        df["y_true"] = pd.to_numeric(df.get("y_true"), errors="coerce")
        df = df.dropna(subset=["pd_final"])

        base_rate = (
            float(df["y_true"].mean())
            if "y_true" in df and df["y_true"].notna().any()
            else None
        )
        avg_pd = float(df["pd_final"].mean()) if not df.empty else 0.0
        calibration_error = abs(avg_pd - base_rate) if base_rate is not None else None
        auc = None
        if roc_auc_score is not None and "y_true" in df and df["y_true"].nunique() > 1:
            try:
                auc = float(roc_auc_score(df["y_true"], df["pd_final"]))
            except Exception:
                auc = None

        logs = [_normalize_risk_log(r) for r in _fetch_recent_risk_logs(limit=500)]
        logs_df = pd.DataFrame(logs)
        risk_event_rate = 0.0
        if not logs_df.empty:
            logs_df["scored_at"] = pd.to_datetime(
                logs_df["scored_at"], errors="coerce", utc=True
            )
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)
            risk_event_rate = float((logs_df["scored_at"] >= cutoff).sum())

        return jsonify(
            {
                "has_data": True,
                "auc": auc,
                "mean_pd": avg_pd,
                "observed_default_rate": base_rate,
                "calibration_error": calibration_error,
                "drift_proxy": float(df["pd_final"].std(ddof=0) * 100),
                "red_rate": float((df["risk_band"] == "RED").mean() * 100),
                "yellow_rate": float((df["risk_band"] == "YELLOW").mean() * 100),
                "green_rate": float((df["risk_band"] == "GREEN").mean() * 100),
                "events_last_hour": risk_event_rate,
                "sample_size": int(len(df)),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-health/timeline", methods=["GET"])
def model_health_timeline():
    try:
        hours = int(request.args.get("hours", 24))
        logs = [_normalize_risk_log(r) for r in _fetch_recent_risk_logs(limit=2000)]
        df = pd.DataFrame(logs)
        if df.empty:
            return jsonify({"items": []})

        df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        df = df[df["scored_at"] >= cutoff]
        if df.empty:
            return jsonify({"items": []})

        df["hour"] = df["scored_at"].dt.floor("h")
        grouped = (
            df.groupby("hour")
            .agg(
                events=("customer_id", "count"),
                avg_pd=("pd_ensemble", "mean"),
                red_events=("risk_band", lambda x: int((x == "RED").sum())),
            )
            .reset_index()
            .sort_values("hour")
        )

        items = []
        for _, row in grouped.iterrows():
            items.append(
                {
                    "time": row["hour"].isoformat(),
                    "events": int(row["events"]),
                    "avg_pd": float(row["avg_pd"]),
                    "red_events": int(row["red_events"]),
                }
            )
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/config/runtime", methods=["GET"])
def runtime_config():
    return jsonify(
        {
            "risk_thresholds": {
                "green_lt": float(banding_config.get("green", 15.0)) / 100.0,
                "red_gte": float(banding_config.get("red", 50.0)) / 100.0,
            },
            "service": {
                "version": "3.0.0-turbo",
                "dashboard_table": "dashboard_customers",
                "risk_log_table": "risk_scores",
            },
        }
    )


@app.route("/customers", methods=["GET"])
def get_customers():
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    band = request.args.get("risk_band")
    query = (
        supabase.table("dashboard_customers")
        .select("*")
        .range(offset, offset + limit - 1)
    )
    if band:
        query = query.eq("risk_band", band.upper())
    res = query.execute()
    return jsonify(res.data)


@app.route("/predict", methods=["POST"])
def predict():
    req = request.json
    try:

        def d(k):
            return float(req.get(k, 0.0) or 0.0)

        # Turbo Feature Engineering
        f1 = d("total_monthly_emi_amount") / (d("total_salary_credit_30d") + 1)
        f2 = 1 - (d("savings_balance_current") / (d("savings_balance_60d_ago") + 1))
        f3 = abs(d("salary_credit_date_m1") - d("expected_salary_date")) / 10
        f5 = d("auto_debit_failure_count_30d") / 5
        f6 = d("lending_app_transaction_count_30d") / 10
        f14 = d("total_loan_outstanding") / (d("total_credit_limit") + 1)

        vec = {
            "stress_f1": np.clip(f1, 0, 1),
            "stress_f2": np.clip(f2, 0, 1),
            "stress_f3": np.clip(f3, 0, 1),
            "stress_f5": np.clip(f5, 0, 1),
            "stress_f6": np.clip(f6, 0, 1),
            "stress_f14": np.clip(f14, 0, 1),
            "vintage": d("customer_vintage_months") / 240.0,
            "age": d("age") / 75.0,
            "income_vol": d("income_volatility_ratio_3m"),
            "overdraft": d("overdraft_days_30d") / 30.0,
        }
        X_df = pd.DataFrame([vec])[FEATURE_NAMES]

        # Stacking Prediction
        p_xgb = m_xgb.predict_proba(X_df)[:, 1][0]
        p_lgbm = m_lgbm.predict_proba(X_df)[:, 1][0]
        X_meta = m_scaler.transform(pd.DataFrame({"xgb": [p_xgb], "lgbm": [p_lgbm]}))
        p_final = float(m_meta.predict_proba(X_meta)[:, 1][0])

        t_g = banding_config["green"] / 100
        t_r = banding_config["red"] / 100
        band = "GREEN" if p_final < t_g else ("YELLOW" if p_final < t_r else "RED")

        customer_id = str(
            req.get("customer_id") or req.get("external_customer_id") or "adhoc"
        )
        _persist_risk_log(customer_id, p_xgb, p_lgbm, p_final, band, t_g, t_r)

        return jsonify(
            {
                "pd_final": round(p_final, 4),
                "pd_xgb": round(float(p_xgb), 4),
                "pd_lgbm": round(float(p_lgbm), 4),
                "risk_band": band,
                "version": "3.0.0-turbo",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
