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
        "metrics": {},  # Dict[cid: str, List[Dict[str, object]]]
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
        cid = request.args.get("cid")
        if cid:
            client_metrics = state["metrics"].get(cid, [])
            if not client_metrics:
                return jsonify({}), 204
            return jsonify(client_metrics[-1])
        else:
            # Return latest global metrics (aggregate or from coordinator)
            all_metrics = []
            for mlist in state["metrics"].values():
                all_metrics.extend(mlist)
            if not all_metrics:
                return jsonify({}), 204
            return jsonify(all_metrics[-1])

    @app.get("/api/metrics/history")
    def metrics_history():
        cid = request.args.get("cid")
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        if cid:
            client_metrics = state["metrics"].get(cid, [])
            items = client_metrics[offset:offset+limit]
            total = len(client_metrics)
        else:
            # Return all metrics combined
            all_metrics = []
            for mlist in state["metrics"].values():
                all_metrics.extend(mlist)
            items = all_metrics[offset:offset+limit]
            total = len(all_metrics)
        return jsonify({
            "items": items,
            "total": total,
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
        cid = str(payload.get("cid", "global"))
        item = {
            "cid": cid,
            "round": int(payload.get("round", 0)),
            "loss": float(payload.get("loss", 0.0)),
            "accuracy": float(payload.get("accuracy", 0.0)),
            "val_auc": float(payload.get("val_auc", 0.0)),
            "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        }
        metrics: Dict[str, List[Dict[str, object]]] = state["metrics"]  # type: ignore
        if cid not in metrics:
            metrics[cid] = []
        metrics[cid].append(item)
        # Keep bounded for memory
        if len(metrics[cid]) > 10000:
            del metrics[cid][:5000]
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


