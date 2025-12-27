# FactCheck-MM Documentation

Comprehensive documentation for the FactCheck-MM multimodal fact-checking system with sarcasm detection and paraphrasing capabilities.

## 📚 Documentation Structure

This documentation is organized for both **research understanding** and **practical implementation**, suitable for academic publication and production deployment.

### Quick Navigation

| Document | Purpose | Target Audience |
|----------|---------|-----------------|
| [Installation Guide](installation.md) | Environment setup and dependency installation | Developers, Researchers |
| [Usage Guide](usage.md) | Running experiments and inference | Researchers, ML Engineers |
| [API Reference](api_reference.md) | REST API endpoints and integration | Engineers, API Users |
| [Model Architectures](model_architectures.md) | Technical architecture details | Researchers, Architects |
| [Dataset Information](dataset_info.md) | Dataset descriptions and statistics | Researchers, Data Scientists |

## 🎯 Project Overview

FactCheck-MM is an end-to-end multimodal fact-checking system that addresses three interconnected challenges:

1. **Sarcasm Detection**: Multimodal sarcasm identification across text, audio, image, and video
2. **Paraphrasing**: Neural paraphrase generation with sarcasm-aware reformulation
3. **Fact Verification**: Evidence retrieval and claim verification with stance detection

### Key Innovations

- **Unified Multimodal Backbone**: Shared encoder architecture (RoBERTa, Wav2Vec2, ViT) across tasks
- **Sarcasm-Aware Pipeline**: Integrated sarcasm detection for robust fact-checking
- **Hybrid Retrieval**: Combined dense (DPR) and sparse (BM25) evidence retrieval
- **Production-Ready**: Docker deployment, FastAPI serving, comprehensive monitoring

## 🔬 Research Context

This system is designed for **reproducible research** and **production deployment**:

- **Academic Focus**: Detailed architecture descriptions, experimental protocols, dataset analysis
- **Engineering Focus**: Scalable deployment, API serving, monitoring infrastructure
- **Reproducibility**: Version-pinned dependencies, deterministic training, comprehensive logging

## 📖 Documentation Roadmap

### For Researchers

1. Start with [Dataset Information](dataset_info.md) to understand data characteristics
2. Review [Model Architectures](model_architectures.md) for technical implementation details
3. Follow [Usage Guide](usage.md) for running experiments
4. Refer to supplementary materials for ablation studies and hyperparameter tuning

### For Practitioners

1. Begin with [Installation Guide](installation.md) for environment setup
2. Follow [Usage Guide](usage.md) for inference and API usage
3. Consult [API Reference](api_reference.md) for integration details
4. Review deployment configurations in `deployment/docker/`

### For Production Deployment

1. Complete [Installation Guide](installation.md) production setup
2. Export models using `deployment/scripts/model_export.py`
3. Deploy with Docker following `deployment/docker/README.md`
4. Monitor using tools described in `deployment/monitoring/`

## 🎓 Citation

If you use FactCheck-MM in your research, please cite:

@software{factcheck_mm_2024,
title={FactCheck-MM: A Multimodal Fact-Checking System with Sarcasm Detection},
author={FactCheck-MM Team},
year={2024},
url={https://github.com/factcheck-mm/FactCheck-MM}
}


## 📋 System Requirements

### Hardware
- **Minimum**: 16GB RAM, 4-core CPU, 50GB storage
- **Recommended**: 32GB RAM, 8-core CPU, RTX 2050+ GPU, 100GB SSD
- **Tested On**: MacBook M2 (16GB), Acer Nitro 5 (RTX 2050, 16GB)

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU)
- Docker 20.10+ (for containerized deployment)

## 🤝 Contributing

For contribution guidelines, see `CONTRIBUTING.md` in the project root.

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

## 🔗 Additional Resources

- **Project Repository**: [GitHub](https://github.com/factcheck-mm)
- **API Documentation**: [Interactive Swagger UI](http://localhost:8000/docs)
- **Research Paper**: [ArXiv](https://arxiv.org) *(to be published)*
- **Demo**: [Live Demo](https://factcheck-mm.demo) *(coming soon)*

## 📞 Support

For questions, issues, or collaboration inquiries:
- Open an issue on GitHub
- Email: support@factcheck-mm.org
- Research inquiries: research@factcheck-mm.org

---

**Last Updated**: December 2024  
**Documentation Version**: 1.0.0  
**Project Version**: 1.0.0
