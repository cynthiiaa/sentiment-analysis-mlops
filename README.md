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

## 🤖 Automated Model Validation

The project includes daily automated model validation that:
- Checks model performance against thresholds (F1 > 0.90)
- Detects data drift using statistical tests
- Runs performance benchmarks
- Generates HTML validation reports

This runs automatically every day at midnight or can be triggered manually in GitHub Actions.

## 🏗️ Project Structure

```
sentiment-analysis-mlops/
├── app/gradio_app.py          # Web interface
├── src/
│   ├── api/inference.py       # REST API
│   ├── models/                # ML models & registry
│   └── monitoring/            # Drift detection & metrics
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Model evaluation
│   ├── deploy.py             # Deployment automation
│   └── generate_report.py    # Report generation
├── configs/
│   ├── training_config.yaml  # Training setup
│   └── deployment_config.yaml # Deploy settings
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # API tests
│   └── performance/          # Latency tests
├── docker/
│   ├── docker-compose.yml    # Multi-service deployment
│   ├── Dockerfile            # Container definition
│   ├── prometheus.yml        # Metrics config
│   └── alerts.yml           # Alert rules
├── .github/workflows/        # CI/CD pipelines
│   ├── model_validation.yml # Daily automated validation
│   ├── ci.yml               # Continuous integration
│   └── cd.yml               # Continuous deployment
├── notebooks/                # Exploration notebooks
├── data/                     # Sample datasets
└── requirements/             # Dependencies
```

## 🎓 Learning Path for MLOps Engineers

### For Beginners - Start Here!

If you're new to MLOps, follow this learning path to understand the project:

#### 📚 Week 1: Understanding the Basics

1. **Start with the notebook** (`notebooks/01_exploration.ipynb`)

   - Understand the data and model
   - See how sentiment analysis works

2. **Run the Gradio app** (`app/gradio_app.py`)

   - See the end-user experience
   - Test different inputs

3. **Explore the model code** (`src/models/sentiment_model.py`)
   - Understand model loading and inference
   - Learn about tokenization and predictions

#### 🚀 Week 2: Training & Experimentation

4. **Study the training config** (`configs/training_config.yaml`)

   - Learn about hyperparameters
   - Understand dataset configuration

5. **Run training** (`scripts/train.py`)

   - Train your first model
   - Monitor with MLflow (http://localhost:5003)

6. **Evaluate models** (`scripts/evaluate.py`)
   - Compare model performance
   - Understand metrics

#### 🔧 Week 3: APIs & Testing

7. **Build the API** (`src/api/inference.py`)

   - Learn FastAPI basics
   - Implement endpoints

8. **Write tests** (`tests/`)
   - Unit tests (`tests/unit/test_model.py`)
   - Integration tests (`tests/integration/test_api.py`)
   - Performance tests (`tests/performance/test_latency.py`)

#### 📊 Week 4: Monitoring & Production

9. **Add monitoring** (`src/monitoring/`)

   - Drift detection (`drift_detection.py`)
   - Metrics collection (`metrics.py`)

10. **Deploy with Docker** (`docker/`)

    - Understand containerization
    - Multi-service orchestration

11. **Set up CI/CD** (`.github/workflows/`)
    - Automated testing
    - Model validation
    - Deployment pipelines

### 🚀 For Experienced Engineers - Quick Setup

1. **Fork & customize configs** → `configs/`
2. **Train your model** → `make train`
3. **Deploy services** → `docker-compose up`
4. **Monitor** → Check Prometheus/Grafana dashboards
5. **Iterate** → Use MLflow for experiment tracking

## 📖 Full Documentation

For detailed documentation including API reference, configuration options, monitoring setup, and deployment guides:

**👉 [See DOCS.md](./DOCS.md)**

## 🔧 Optional Setup

### Code Coverage (Codecov)

To enable code coverage reports in CI/CD:

1. Sign up at [codecov.io](https://codecov.io)
2. Add your repository
3. Copy the token
4. Add it as `CODECOV_TOKEN` in GitHub Settings → Secrets

_Note: Coverage reporting will work without this, but you'll see rate limit warnings._

## 🆘 Need Help?

**Common Issues:**

- **Port conflicts**: Change ports in `docker-compose.yml`
- **Memory issues**: Reduce batch size in `configs/training_config.yaml`
- **Docker problems**: Try `docker-compose down` then `up` again

**Get Support:**

- 📖 Check [DOCS.md](./DOCS.md) for detailed guides
- 🐛 Report issues in GitHub Issues
- 💡 See example code in `DOCS.md`
