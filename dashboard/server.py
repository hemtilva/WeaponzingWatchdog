"""Lightweight local dashboard server for interactive market simulation and visualization."""

import http.server
import json
import os
import sys
import urllib.parse
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.market import MarketConfig
from src.engine.rules import RuleType
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


DASHBOARD_DIR = Path(__file__).resolve().parent


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Handles static web assets and simulation API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/simulate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            rule_str = data.get("rule", "mean")
            delta = float(data.get("delta", 0.20))
            t_ban = int(data.get("t_ban", 3))
            k0 = float(data.get("k0", 1.0))
            phi = float(data.get("phi", 0.10))
            steps = int(data.get("steps", 30_000))
            warmup = int(data.get("warmup", 5_000))
            agent_type = str(data.get("agent_type", "adaptive"))
            seed = int(data.get("seed", 42))

            rule_map = {
                "mean": RuleType.MEAN_RATIO,
                "median": RuleType.MEDIAN_RATIO,
                "cap": RuleType.PRICE_CAP,
            }
            rule_type = rule_map.get(rule_str, RuleType.MEAN_RATIO)

            m_cfg = MarketConfig(
                num_firms=3,
                price_sensitivity=0.25,
                fixed_cost=phi,
                initial_capital=k0,
                ban_duration=t_ban,
                threshold_delta=delta,
            )
            sim_cfg = SimulationConfig(
                market=m_cfg,
                total_steps=steps,
                warmup_steps=warmup,
                episode_length=500,
                rule_type=rule_type,
                rule_param=delta if rule_str != "cap" else 1.8,
                record_trace=True,
                trace_steps=200,
                agent_type=agent_type,
                seed=seed,
            )

            trainer = MultiAgentTrainer(sim_cfg)
            results = trainer.run()

            response = {
                "weaponization_rate": results.weaponization_rate,
                "collusion_index": results.collusion_index,
                "social_welfare_loss": results.social_welfare_loss,
                "bankruptcy_rate": results.bankruptcy_rate,
                "mean_prices": [float(p) for p in results.mean_prices],
                "total_fresh_bans": results.total_fresh_bans,
                "induced_bans": results.induced_bans,
                "exploit_events": results.exploit_events or [],
                "trace": results.trace_data or [],
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")


def start_server(port: int = 8000):
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, DashboardHandler)
    print(f"\n=======================================================")
    print(f"Weaponizing the Watchdog: Interactive Web Dashboard")
    print(f"Server running at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port)

