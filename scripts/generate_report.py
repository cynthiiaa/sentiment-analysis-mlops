#!/usr/bin/env python3
"""
Generate HTML reports for model validation and evaluation results.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def load_evaluation_results(
    results_path: str = "results/evaluation_results.json",
) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    if Path(results_path).exists():
        with open(results_path, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: No evaluation results found at {results_path}")
        return {}


def load_performance_data() -> Dict[str, Any]:
    """Load performance test data if available"""
    performance_data = {}

    # Try to load performance analysis CSV if it exists
    perf_path = Path("results/performance_analysis.csv")
    if perf_path.exists():
        import pandas as pd

        df = pd.read_csv(perf_path)
        performance_data = {
            "avg_latency": df["latency_ms"].mean() if "latency_ms" in df else None,
            "p95_latency": df["latency_ms"].quantile(0.95) if "latency_ms" in df else None,
            "throughput": df["throughput"].mean() if "throughput" in df else None,
        }

    return performance_data


def generate_html_report(
    evaluation_results: Dict[str, Any], performance_data: Dict[str, Any], output_path: str
) -> None:
    """Generate HTML report from results"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract metrics
    metrics = evaluation_results.get("metrics", {})

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Validation Report - {timestamp}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        .header .timestamp {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}

        .content {{
            padding: 40px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .metric-card {{
            background: #f7f7f7;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .metric-card .label {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}

        .metric-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }}

        .metric-card .unit {{
            font-size: 0.9rem;
            color: #999;
            margin-left: 5px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }}

        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
        }}

        .status-passed {{
            background: #10b981;
            color: white;
        }}

        .status-failed {{
            background: #ef4444;
            color: white;
        }}

        .status-warning {{
            background: #f59e0b;
            color: white;
        }}

        .chart-container {{
            background: #f7f7f7;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .distribution-bar {{
            display: flex;
            height: 40px;
            border-radius: 5px;
            overflow: hidden;
            margin: 20px 0;
        }}

        .distribution-positive {{
            background: #10b981;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}

        .distribution-negative {{
            background: #ef4444;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}

        .info-item {{
            background: #f7f7f7;
            padding: 15px;
            border-radius: 8px;
        }}

        .info-item strong {{
            color: #667eea;
        }}

        .footer {{
            background: #f7f7f7;
            padding: 20px;
            text-align: center;
            color: #666;
        }}

        .performance-metrics {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}

        .performance-metrics .metric-card {{
            background: white;
            border-left-color: #fcb69f;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Model Validation Report</h1>
            <div class="timestamp">Generated on {timestamp}</div>
        </div>

        <div class="content">
            <!-- Overall Status -->
            <div class="section">
                <h2>Overall Status</h2>
                <div style="text-align: center; padding: 20px;">
"""

    # Determine overall status
    f1_score = metrics.get("f1_score", 0)
    if f1_score >= 0.90:
        status = "PASSED"
        status_class = "status-passed"
        emoji = "✅"
    elif f1_score >= 0.80:
        status = "WARNING"
        status_class = "status-warning"
        emoji = "⚠️"
    else:
        status = "FAILED"
        status_class = "status-failed"
        emoji = "❌"

    html_content += f"""
                    <div style="font-size: 3rem; margin-bottom: 10px;">{emoji}</div>
                    <span class="status-badge {status_class}">{status}</span>
                    <p style="margin-top: 15px; color: #666;">
                        Model F1 Score: <strong>{f1_score:.4f}</strong>
                    </p>
                </div>
            </div>

            <!-- Model Performance Metrics -->
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Accuracy</div>
                        <div class="value">{metrics.get('accuracy', 0):.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">F1 Score</div>
                        <div class="value">{metrics.get('f1_score', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Precision</div>
                        <div class="value">{metrics.get('precision', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Recall</div>
                        <div class="value">{metrics.get('recall', 0):.4f}</div>
                    </div>
                </div>
            </div>
"""

    # Add class distribution if available
    if "class_distribution" in evaluation_results:
        dist = evaluation_results["class_distribution"]
        pred_pos = dist.get("predicted", {}).get("positive", 0)
        pred_neg = dist.get("predicted", {}).get("negative", 0)
        total = pred_pos + pred_neg

        if total > 0:
            pos_pct = (pred_pos / total) * 100
            neg_pct = (pred_neg / total) * 100

            html_content += f"""
            <!-- Class Distribution -->
            <div class="section">
                <h2>Prediction Distribution</h2>
                <div class="chart-container">
                    <p style="margin-bottom: 10px;">Predicted Sentiment Distribution</p>
                    <div class="distribution-bar">
                        <div class="distribution-positive" style="width: {pos_pct}%">
                            Positive {pos_pct:.1f}%
                        </div>
                        <div class="distribution-negative" style="width: {neg_pct}%">
                            Negative {neg_pct:.1f}%
                        </div>
                    </div>
                    <div class="info-grid" style="margin-top: 20px;">
                        <div class="info-item">
                            <strong>Total Samples:</strong> {total}
                        </div>
                        <div class="info-item">
                            <strong>Correct Predictions:</strong> {evaluation_results.get('correct_predictions', 'N/A')}
                        </div>
                    </div>
                </div>
            </div>
"""

    # Add performance metrics if available
    if performance_data:
        html_content += """
            <!-- Performance Metrics -->
            <div class="section">
                <h2>System Performance</h2>
                <div class="performance-metrics">
                    <div class="metrics-grid">
"""

        if performance_data.get("avg_latency"):
            html_content += f"""
                        <div class="metric-card">
                            <div class="label">Avg Latency</div>
                            <div class="value">{performance_data['avg_latency']:.2f}<span class="unit">ms</span></div>
                        </div>
"""

        if performance_data.get("p95_latency"):
            html_content += f"""
                        <div class="metric-card">
                            <div class="label">P95 Latency</div>
                            <div class="value">{performance_data['p95_latency']:.2f}<span class="unit">ms</span></div>
                        </div>
"""

        if performance_data.get("throughput"):
            html_content += f"""
                        <div class="metric-card">
                            <div class="label">Throughput</div>
                            <div class="value">{performance_data['throughput']:.1f}<span class="unit">req/s</span></div>
                        </div>
"""

        html_content += """
                    </div>
                </div>
            </div>
"""

    html_content += """
            <!-- Test Information -->
            <div class="section">
                <h2>Test Configuration</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Dataset:</strong> GLUE SST-2
                    </div>
                    <div class="info-item">
                        <strong>Split:</strong> Validation
                    </div>
                    <div class="info-item">
                        <strong>Model:</strong> DistilBERT
                    </div>
                    <div class="info-item">
                        <strong>Threshold:</strong> F1 ≥ 0.90
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Generated by MLOps Sentiment Analysis Platform</p>
            <p>Report Version 1.0.0</p>
        </div>
    </div>
</body>
</html>
"""

    # Write the report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✅ Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate validation report")
    parser.add_argument(
        "--results",
        type=str,
        default="results/evaluation_results.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"reports/validation_{datetime.now().strftime('%Y%m%d')}.html",
        help="Output path for HTML report",
    )

    args = parser.parse_args()

    # Load results
    evaluation_results = load_evaluation_results(args.results)
    performance_data = load_performance_data()

    # Generate report
    generate_html_report(evaluation_results, performance_data, args.output)


if __name__ == "__main__":
    main()
