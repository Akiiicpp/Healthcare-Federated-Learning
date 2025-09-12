# Healthcare Federated Learning Platform

A federated learning platform enabling multiple hospitals to collaboratively train AI models for cancer detection while maintaining strict data privacy. The system processes medical imaging data locally at each hospital and only shares encrypted model parameters, creating a powerful diagnostic tool trained on diverse, large-scale datasets.

## 🏥 Overview

This project simulates a federated learning environment where multiple hospitals collaborate to train a cancer detection model without sharing raw patient data. Each hospital trains locally on their own data and only shares model updates with a central coordinator.

## ✨ Features

- **Privacy-Preserving**: No raw data leaves individual hospitals
- **Federated Learning**: Collaborative training using Flower framework
- **Medical AI**: Lightweight ResNet-18 for 2D medical image classification
- **Docker Support**: Easy deployment and scaling
- **Real-time Monitoring**: Live training progress tracking
- **Synthetic Data**: Safe demo with generated medical images

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop (for containerized deployment)
- 4GB+ RAM recommended

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/healthcare-federated-learning.git
   cd healthcare-federated-learning
   ```

2. **Set up environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run local training (Phase 1)**
   ```bash
   python run_local_training.py
   ```

4. **Run federated learning (Phase 2)**
   ```bash
   ./run_federated.sh --clients 3 --rounds 5
   ```

### Docker Deployment

1. **Quick demo (3 clients, 3 rounds)**
   ```bash
   ./run_docker.sh --scenario quick
   ```

2. **Large demo (5 clients, 10 rounds)**
   ```bash
   ./run_docker.sh --scenario large
   ```

## 📁 Project Structure

```
federated_healthcare_demo/
├── coordinator/              # Central server
│   ├── server.py            # Flower server
│   ├── aggregator.py        # Model aggregation
│   └── privacy_auditor.py   # Privacy tracking
├── hospital_client/         # Hospital nodes
│   ├── client.py            # Flower client
│   ├── trainer.py           # Local training
│   └── data_loader.py       # Data handling
├── models/                  # ML models
│   ├── cnn_model.py         # ResNet-18 model
│   └── privacy_utils.py     # DP utilities
├── dashboard/               # Monitoring (Phase 3)
├── data/                    # Dataset storage
├── requirements.txt         # Dependencies
├── run_local_training.py    # Phase 1 script
├── run_federated.sh         # Phase 2 script
├── run_docker.sh            # Phase 4 script
├── docker-compose.yml       # Docker orchestration
├── Dockerfile.coordinator   # Server container
├── Dockerfile.client        # Client container
└── COMMANDS_REFERENCE.txt   # Complete command guide
```

## 🎯 Phases

- **Phase 1**: Local training sanity check
- **Phase 2**: Federated learning with multiple hospitals
- **Phase 3**: Real-time dashboard (planned)
- **Phase 4**: Docker deployment and scaling
- **Phase 5**: Advanced features (DP, robust aggregation)

## 📊 Expected Results

### Local Training
- **Accuracy**: 100% on synthetic data
- **Time**: ~30 seconds
- **Output**: Saved model checkpoint

### Federated Learning
- **Accuracy**: 33% → 100% over rounds
- **Time**: 1-5 minutes (depending on rounds)
- **Clients**: 3-10+ hospitals
- **Privacy**: No data sharing between hospitals

## ��️ Configuration

### Command Line Options

**Federated Learning:**
```bash
./run_federated.sh --clients 5 --rounds 10 --local-epochs 2 --num-samples 800 --batch-size 32
```

**Docker Deployment:**
```bash
./run_docker.sh --scenario large --clients 5 --rounds 10
```

### Environment Variables

- `CLIENT_ID`: Hospital identifier
- `NUM_SAMPLES`: Dataset size per client
- `BATCH_SIZE`: Training batch size
- `SEED`: Random seed for reproducibility

## 🔧 Troubleshooting

### Common Issues

1. **Port 8080 in use**
   ```bash
   lsof -ti:8080 | xargs kill -9
   ```

2. **Docker not running**
   - Start Docker Desktop
   - Check with `docker info`

3. **Permission denied**
   ```bash
   chmod +x run_federated.sh run_docker.sh
   ```

4. **Virtual environment issues**
   ```bash
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## �� Documentation

- **Complete Command Reference**: `COMMANDS_REFERENCE.txt`
- **Docker Setup Guide**: `README-Docker.md`
- **API Documentation**: `docs/` (planned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flower](https://flower.dev/) - Federated learning framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Streamlit](https://streamlit.io/) - Dashboard framework

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note**: This is a research/demo project using synthetic data. For production use with real medical data, ensure compliance with healthcare regulations (HIPAA, GDPR, etc.).
