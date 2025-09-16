from typing import Dict, List, Optional
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # In-memory store for quick start; replace with DB later
    state: Dict[str, object] = {
        "round": 0,
        "metrics": [],  # List[Dict[str, object]]
        "clients": [],  # List[Dict[str, object]]
    }

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})

    @app.get("/api/rounds/current")
    def current_round():
        return jsonify({"round": state["round"]})

    @app.get("/api/metrics/latest")
    def latest_metrics():
        metrics: List[Dict[str, object]] = state["metrics"]  # type: ignore
        if not metrics:
            return jsonify({}), 204
        return jsonify(metrics[-1])

    @app.get("/api/metrics/history")
    def metrics_history():
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        metrics: List[Dict[str, object]] = state["metrics"]  # type: ignore
        return jsonify({
            "items": metrics[offset:offset+limit],
            "total": len(metrics),
            "offset": offset,
            "limit": limit,
        })

    @app.get("/api/clients")
    def list_clients():
        clients: List[Dict[str, object]] = state["clients"]  # type: ignore
        return jsonify({"items": clients, "total": len(clients)})

    # Internal endpoints to be called by coordinator during training
    @app.post("/api/internal/round")
    def set_round():
        payload = request.get_json(silent=True) or {}
        state["round"] = int(payload.get("round", state["round"]))
        return jsonify({"ok": True, "round": state["round"]})

    @app.post("/api/internal/metrics")
    def add_metrics():
        payload = request.get_json(silent=True) or {}
        item = {
            "round": int(payload.get("round", 0)),
            "loss": float(payload.get("loss", 0.0)),
            "accuracy": float(payload.get("accuracy", 0.0)),
            "val_auc": float(payload.get("val_auc", 0.0)),
            "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        }
        metrics: List[Dict[str, object]] = state["metrics"]  # type: ignore
        metrics.append(item)
        # Keep bounded for memory
        if len(metrics) > 10000:
            del metrics[:5000]
        return jsonify({"ok": True})

    @app.post("/api/internal/clients")
    def upsert_client():
        payload = request.get_json(silent=True) or {}
        cid = str(payload.get("cid", ""))
        status = str(payload.get("status", "unknown"))
        info = {
            "cid": cid,
            "status": status,
            "last_seen": datetime.utcnow().isoformat() + "Z",
        }
        clients: List[Dict[str, object]] = state["clients"]  # type: ignore
        found = False
        for i, c in enumerate(clients):
            if c.get("cid") == cid:
                clients[i] = {**c, **info}
                found = True
                break
        if not found:
            clients.append(info)
        return jsonify({"ok": True})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)


