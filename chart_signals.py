"""
Render a local HTML chart of recent BTC/ETH closes colored by live signal.
"""

from __future__ import annotations

import json
from html import escape

import asset_config as ac
import predict_latest as pl
import train as tr
from prepare import add_cross_asset_context_features, add_price_features, download_symbol_prices

DEFAULT_LOOKBACK_BARS = 180
SIGNAL_COLORS = {
    "no_trade": "#9ca3af",
    "long": "#15803d",
    "short": "#b91c1c",
}


def get_env_int(name: str, default: int) -> int:
    value = tr.os.getenv(name)
    return int(value) if value is not None else default


def build_chart_rows(lookback_bars: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    asset_key = ac.get_asset_key()
    tr.set_seed(tr.get_env_int("AR_SEED", tr.SEED))
    raw_prices = download_symbol_prices(asset_key)
    live_features = add_price_features(raw_prices)
    live_features = add_cross_asset_context_features(live_features, asset_key)

    long_model, long_state = tr.fit_model("long")
    short_model, short_state = tr.fit_model("short")
    long_feature_names = list(long_model.feature_names)
    short_feature_names = list(short_model.feature_names)
    required_feature_names = sorted(set(long_feature_names) | set(short_feature_names))

    train_frame_long = long_state["splits"]["train"].frame
    train_frame_short = short_state["splits"]["train"].frame
    scored = live_features.dropna(subset=required_feature_names).tail(lookback_bars).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx in range(len(scored)):
        row = scored.iloc[[idx]]
        long_vector, snapshot = pl.score_row(long_feature_names, train_frame_long, row)
        short_vector, _ = pl.score_row(short_feature_names, train_frame_short, row)
        long_probability = float(tr.sigmoid(long_vector @ long_model.weights)[0])
        short_probability = float(tr.sigmoid(short_vector @ short_model.weights)[0])
        signal, decision_info = pl.choose_final_signal(
            asset_key,
            long_probability,
            float(long_model.threshold),
            short_probability,
            float(short_model.threshold),
            snapshot,
        )
        rows.append(
            {
                "date": row["date"].iloc[0].strftime("%Y-%m-%d"),
                "close": round(float(row["close"].iloc[0]), 2),
                "signal": signal,
                "decision_reason": str(decision_info["reason"]),
                "long_probability": round(long_probability, 4),
                "short_probability": round(short_probability, 4),
                "long_gap": round(long_probability - float(long_model.threshold), 4),
                "short_gap": round(short_probability - float(short_model.threshold), 4),
                "ret_7": round(float(snapshot.get("ret_7", 0.0)), 4),
                "drawdown_7": round(float(snapshot.get("drawdown_7", 0.0)), 4),
                "volume_vs_7": round(float(snapshot.get("volume_vs_7", 0.0)), 4),
                "rsi_7": round(float(snapshot.get("rsi_7", 0.0)), 2),
            }
        )

    meta = {
        "asset_key": asset_key,
        "symbol": ac.get_asset_symbol(asset_key),
        "lookback_bars": lookback_bars,
        "latest_date": rows[-1]["date"] if rows else None,
        "signal_colors": SIGNAL_COLORS,
    }
    return rows, meta


def build_html(rows: list[dict[str, object]], meta: dict[str, object]) -> str:
    payload = json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False)
    title = f"{meta['symbol']} Weekly Signal Chart"
    legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{escape(color)}"></span>{escape(name)}</span>'
        for name, color in SIGNAL_COLORS.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f6f4ed 0%, #ebe5d7 100%);
      color: #1f2937;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #e9e1d3;
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 18px 60px rgba(31,41,55,0.08);
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin: 12px 0 18px;
      font-size: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 4px;
      display: inline-block;
    }}
    #chart {{
      width: 100%;
      overflow-x: auto;
      border-top: 1px solid #efe6d8;
      padding-top: 10px;
    }}
    svg {{
      display: block;
      height: 560px;
    }}
    .tooltip {{
      position: fixed;
      display: none;
      pointer-events: none;
      background: rgba(17,24,39,0.94);
      color: white;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 12px;
      white-space: pre-line;
      transform: translate(12px, 12px);
      max-width: 320px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{escape(title)}</h1>
      <div>Latest bar: {escape(str(meta["latest_date"]))}. Signals distinguish `long`, `short`, and `no_trade`.</div>
      <div class="legend">{legend}</div>
      <div id="chart"></div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const payload = {payload};
    const rows = payload.rows;
    const colors = payload.meta.signal_colors;
    const chart = document.getElementById('chart');
    const tooltip = document.getElementById('tooltip');
    const width = Math.max(2200, rows.length * 12);
    const height = 560;
    const leftPad = 56;
    const rightPad = 24;
    const topPad = 20;
    const priceHeight = 420;
    const axisTop = 455;
    const closes = rows.map(r => r.close);
    const minClose = Math.min(...closes);
    const maxClose = Math.max(...closes);
    const range = Math.max(maxClose - minClose, 1);
    const innerWidth = width - leftPad - rightPad;
    const barWidth = innerWidth / rows.length;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));

    for (let i = 0; i < 5; i += 1) {{
      const y = topPad + (priceHeight / 4) * i;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', String(leftPad));
      line.setAttribute('x2', String(width - rightPad));
      line.setAttribute('y1', String(y));
      line.setAttribute('y2', String(y));
      line.setAttribute('stroke', '#d7d0c4');
      line.setAttribute('stroke-dasharray', '3 5');
      svg.appendChild(line);
    }}

    rows.forEach((row, index) => {{
      const x = leftPad + index * barWidth;
      const normalized = (row.close - minClose) / range;
      const barHeight = Math.max(2, normalized * (priceHeight - 8));
      const y = topPad + priceHeight - barHeight;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(Math.max(1, barWidth - 1)));
      rect.setAttribute('height', String(barHeight));
      rect.setAttribute('fill', colors[row.signal] || '#9ca3af');
      rect.setAttribute('rx', '1.5');
      rect.addEventListener('mousemove', (event) => {{
        tooltip.style.display = 'block';
        tooltip.style.left = `${{event.clientX}}px`;
        tooltip.style.top = `${{event.clientY}}px`;
        tooltip.textContent = `${{row.date}}\\nclose=${{row.close}}\\nsignal=${{row.signal}}\\nlong_p=${{row.long_probability}}\\nshort_p=${{row.short_probability}}\\nlong_gap=${{row.long_gap}}\\nshort_gap=${{row.short_gap}}\\nreason=${{row.decision_reason}}\\nret_7=${{row.ret_7}}\\ndrawdown_7=${{row.drawdown_7}}\\nvolume_vs_7=${{row.volume_vs_7}}\\nrsi_7=${{row.rsi_7}}`;
      }});
      rect.addEventListener('mouseleave', () => {{
        tooltip.style.display = 'none';
      }});
      svg.appendChild(rect);
    }});

    const tickEvery = Math.max(1, Math.floor(rows.length / 16));
    rows.forEach((row, index) => {{
      if (index % tickEvery !== 0 && index !== rows.length - 1) return;
      const x = leftPad + index * barWidth + barWidth / 2;
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', String(x));
      label.setAttribute('y', String(axisTop + 18));
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('fill', '#6b7280');
      label.setAttribute('font-size', '11');
      label.textContent = row.date;
      svg.appendChild(label);
    }});

    chart.appendChild(svg);
    requestAnimationFrame(() => {{
      chart.scrollLeft = chart.scrollWidth;
    }});
  </script>
</body>
</html>"""


def main() -> None:
    rows, meta = build_chart_rows(get_env_int("AR_CHART_LOOKBACK_BARS", DEFAULT_LOOKBACK_BARS))
    output_path = ac.get_chart_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(rows, meta), encoding="utf-8")
    print(f"Saved chart to: {output_path}")
    print(f"Bars rendered: {len(rows)}")


if __name__ == "__main__":
    main()
