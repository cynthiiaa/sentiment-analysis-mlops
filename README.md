# 🚀 Sentiment Analysis MLOps Platform

A production-ready sentiment analysis system built with modern MLOps practices, featuring real-time inference, monitoring, and CI/CD pipelines.

## ✨ What You Get

- 🤖 **Pre-trained sentiment analysis** ready to use
- 🎨 **Web interface** for easy testing (Gradio)
- 🔧 **REST API** for integration with your apps
- 📊 **MLflow tracking** for experiment management
- 🐳 **Docker deployment** with one command
- 🏋️ **Custom model training** with your own data

## 🎯 Perfect For

- **Data scientists** wanting to deploy ML models quickly
- **Developers** needing sentiment analysis in their applications
- **Students** learning MLOps best practices
- **Teams** building production ML systems

## ⚡ Quick Start

### Option 1: Docker (Recommended - Everything included!)

```bash
# Clone and start everything
git clone <your-repo-url>
cd sentiment-analysis-mlops

# Start all services (takes ~2 minutes first time)
docker-compose -f docker/docker-compose.yml up -d

# 🎉 You now have:
# Web Interface: http://localhost:7860
# REST API: http://localhost:8000
# MLflow UI: http://localhost:5003
```

### Option 2: Local Development

```bash
# Set up Python environment
python -m venv mlops
source mlops/bin/activate  # Windows: mlops\Scripts\activate

# Install dependencies
make install

# Start the web interface
make run-app
# Visit: http://localhost:7860
```

## 🧪 Test It Out

**Web Interface**: Go to http://localhost:7860 and type "I love this product!"

**API Testing**:

```bash
# Test API health
curl http://localhost:8000/health

# Analyze sentiment
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is amazing!"], "return_probabilities": true}'
```

## 🏋️ Train Your Own Model

```bash
# Customize training in configs/training_config.yaml
make train

# Monitor progress at http://localhost:5003 (MLflow UI)
```

## 🏗️ Project Structure

```
sentiment-analysis-mlops/
├── app/gradio_app.py          # Web interface
├── src/api/inference.py       # REST API
├── configs/training_config.yaml # Training setup
├── scripts/train.py           # Training script
├── docker/docker-compose.yml  # Multi-service deployment
└── requirements/              # Dependencies
```

## 📖 Full Documentation

For detailed documentation including API reference, configuration options, monitoring setup, and deployment guides:

**👉 [See DOCS.md](./DOCS.md)**

## 🆘 Need Help?

**Common Issues:**

- **Port conflicts**: Change ports in `docker-compose.yml`
- **Memory issues**: Reduce batch size in `configs/training_config.yaml`
- **Docker problems**: Try `docker-compose down` then `up` again

**Get Support:**

- 📖 Check [DOCS.md](./DOCS.md) for detailed guides
- 🐛 Report issues in GitHub Issues
- 💡 See example code in `DOCS.md`
