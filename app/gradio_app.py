import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from src.models.sentiment_model import SentimentModel, ModelConfig
from src.monitoring.metrics import MetricsCollector, monitor_performance
from src.monitoring.drift_detection import DriftDetector
import json
from datetime import datetime

# initialize
model = SentimentModel()
metrics_collector = MetricsCollector()
drift_detector = DriftDetector()

def analyze_sentiment(text: str, enable_monitoring: bool = True):
    """Main prediction function with monitoring"""
    if not text:
        return "Please enter some text", None, None, None
    
    # make predicition with monitoring
    @monitor_performance
    def predict():
        return model.predict([text])[0]
    
    result, latency = predict()

    if enable_monitoring:
        metrics_collector.track_predicition(result, latency)
        drift_detector.add_prediction(
            result["confidence"],
            result["sentiment"]
        )

    # create confidence visualization
    confidence_fig = create_confidence_chart(result["probabilities"])

    # get metrics summary
    metrics_summary = metrics_collector.get_metrics_summary()

    # check for drift
    drift_status = drift_detector.detect_drift()

    return (
        f"**Sentiment:** {result['sentiment'].upper()}",
        f"**Confidence:** {result['confidence']:.2%}",
        confidence_fig,
        create_metrics_display(metrics_summary, drift_status)
    )

def create_confidence_chart(probabilities):
    """Create confidence visualization"""
    fig = go.Figure(data=[
        go.Bar(
            x=["Negative", "Positive"],
            y=[probabilities["negative"], probabilities["positive"]],
            marker_color=["red", "green"]
        )
    ])
    fig.update_layout(
        title="Sentiment Probabilities",
        yaxis_title="Probability",
        height=300
    )
    return fig

def create_metrics_display(metrics, drift_status):
    """Format metricss for display"""
    if not metrics:
        return "No metrics available yet"
    
    display = f"""
    ### Performance Metrics
    - Total Predictions: {metrics.get("total_predictions",0)}
    - Avg Latency: {metrics.get("avg_latency", 0):.3f}s
    - P95 Latency: {metrics.get("p95_latency", 0):.3f}s
    - Avg Confidence: {metrics.get("avg_confidence", 0):.2%}
    
    ### Drift Detection
    - Status: {"⚠️ DRIFT DETECTED" if drift_status.get("drift_detected") else "✅ No drift"}
    """

    if drift_status.get("p_value"):
        display += f"- P-value: {drift_status['p_value']:.4f}"

    return display

def batch_analysis(file):
    """Analyze batch of texts from uploaded file"""
    df = pd.read_csv(file.name)
    texts = df["text"].tolist()

    results = model.predict(texts)

    # create results dataframe
    results_df = pd.DataFrame(results)

    # create visualizations
    sentiment_dist = results_df["sentiment"].value_counts()

    dist_fig = go.Figure(data=[
        go.Pie(
            labels=sentiment_dist.index,
            values=sentiment_dist.values,
            hole=0.3
        )
    ])
    dist_fig.update_layout(title="Sentiment Distribution")

    conf_fig = go.Figure(data=[
        go.Histogram(
            x=results_df["confidence"],
            nbinsx=20
        )
    ])
    conf_fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count"
    )

    return results_df, dist_fig, conf_fig

# create Gradio interface
with gr.Blocks(title="MLOps Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Production Sentiment Analysis with MLOps

        This application demonstrates production-ready ML depoloyment with:
        - Real-time inference with Hugging Face models
        - Performance monitoring and metrics collection
        - Drift detection and model health checks
        - Batch processing capabilities
        """
    )

    with gr.Tab("Single Prediction"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Enter text for sentiment analysis",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                monitor_checkbox = gr.Checkbox(
                    label="Enable Monitoring",
                    value=True
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

            with gr.Column():
                sentiment_output = gr.Markdown(label="Result")
                confidence_output = gr.Markdown(label="Confidence")
                confidence_chart = gr.Plot(label="Confidence Distribution")

        metrics_display = gr.Markdown(label="System Metrics")

        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=[text_input, monitor_checkbox],
            outputs=[sentiment_output, confidence_output, confidence_chart, metrics_display]
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown(
            """
            Upload a CSV file with a 'text' column for batch sentiment analysis.
            """
        )

        file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        batch_btn = gr.Button("Analyze Batch", variant="primary")

        results_df = gr.DataFrame(label="Results")
        with gr.Row():
            dist_chart = gr.Plot(label="Sentiment Distribution")
            conf_chart = gr.Plot(label="Confidence Distribution")

        batch_btn.click(
            fn=batch_analysis,
            inputs=file_upload,
            outputs=[results_df, dist_chart, conf_chart]
        )

    with gr.Tab("Model Info"):
        gr.Markdown(
            f"""
            ### Model Information
            {
                json.dumps(model.get_model_info(), indent=2)
                }
            

            ### MLOps Features
            - **Model Registry:** MLflow for versioning and deployment
            - **Monitoring:** Prometheus metrics + custom tracking
            - **Drift Detection:** KS-test for distribution shifts
            - **CI/CD:** GitHub Actions for automated testing
            """
        )
        
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)