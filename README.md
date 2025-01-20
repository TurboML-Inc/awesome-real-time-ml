# Awesome Real-Time Machine Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

We at TurboML are democratising real time ML / AI. Here's a curated list of awesome real-time machine learning blogs, videos, tools and platforms, conferences, research papers, etc.

## Table of Contents
- [Tools & Workflow Stages](#tools--workflow-stages)
- [Traditional ML vs Real-Time ML](#traditional-ml-vs-real-time-ml)
- [Real-Time ML Internal Platform Resources](#real-time-ml-internal-platform-resources)
- [Videos](#videos)
- [Vendors / Platforms](#vendors--platforms)
- [Conferences](#conferences)

## Tools & Workflow Stages

1. **Data Ingestion and Streaming**
   - [Apache Kafka](https://kafka.apache.org/)
   - [Apache Flink](https://flink.apache.org/)
   - [Apache Pulsar](https://pulsar.apache.org/)
   - [Amazon Kinesis](https://aws.amazon.com/kinesis/)

2. **Feature Engineering and Feature Store**
   - [Feast](https://feast.dev/)
   - [Tecton](https://www.tecton.ai/)
   - [Amazon SageMaker Feature Store](https://aws.amazon.com/sagemaker/feature-store/)
   - [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore)

3. **Model Development and Training**
   - [TensorFlow](https://www.tensorflow.org/)
   - [PyTorch](https://pytorch.org/)
   - [Scikit-learn](https://scikit-learn.org/)
   - [MLflow](https://mlflow.org/)
   - [Kubeflow](https://www.kubeflow.org/)

4. **Model Deployment and Serving**
   - [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
   - [TorchServe](https://pytorch.org/serve/)
   - [Seldon Core](https://www.seldon.io/solutions/open-source-projects/core)
   - [KServe](https://kserve.github.io/website/)
   - [BentoML](https://www.bentoml.com/)

5. **Real-Time Inference**
   - [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
   - [Apache TVM](https://tvm.apache.org/)
   - [ONNX Runtime](https://onnxruntime.ai/)

6. **Monitoring and Feedback Loop**
   - [Prometheus](https://prometheus.io/)
   - [Grafana](https://grafana.com/)
   - [Evidently AI](https://www.evidentlyai.com/)
   - [WhyLabs](https://whylabs.ai/)

7. **Orchestration and MLOps**
   - [Apache Airflow](https://airflow.apache.org/)
   - [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
   - [MLflow](https://mlflow.org/)
   - [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)

8. **Cloud Platforms**
   - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
   - [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)
   - [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/)

## Traditional ML vs Real-Time ML

| Aspect | Traditional ML | Real-Time ML |
|--------|----------------|--------------|
| **Data Processing** | Processes static, historical datasets in batches. | Continuously ingests and processes streaming data in real-time. |
| **Model Training** | Models are trained offline using complete datasets. | Models are updated incrementally as new data arrives, often using online learning algorithms. |
| **Latency** | Can tolerate higher latency in processing and predictions. | Requires low-latency processing and near-instantaneous predictions. |
| **Scalability** | Typically scales vertically with more powerful hardware. | Often requires horizontal scalability to handle high-volume data streams. |
| **Infrastructure** | Can run on standard computing resources. | Often requires specialized streaming infrastructure like Apache Kafka or Apache Flink. |
| **Adaptability** | Models are less adaptive to changing patterns without manual retraining. | Models can adapt to concept drift and evolving patterns in real-time. |
| **Feature Engineering** | Features are often engineered manually and in advance. | Features may be generated on-the-fly or use automated feature extraction techniques. |
| **Model Deployment** | Models are deployed as static versions, updated periodically. | Models are continuously updated and deployed in a streaming fashion. |
| **Use Cases** | Suitable for predictive analytics, customer segmentation, and offline recommendations. | Ideal for fraud detection, real-time bidding, and personalized recommendations. |
| **Data Volume** | Can work effectively with smaller datasets. | Typically requires larger volumes of data for accurate real-time predictions. |
| **Computational Resources** | Generally less computationally intensive. | Often requires more computational power to process streaming data continuously. |
| **Monitoring** | Periodic model performance checks are sufficient. | Requires continuous monitoring of model performance and data quality. |
| **Feedback Loop** | Feedback is incorporated in batch updates. | Immediate feedback integration for rapid model adjustments. |
| **Complexity** | Generally simpler to implement and maintain. | More complex, requiring specialized knowledge in streaming architectures and online learning algorithms. |
| **Time-to-Insight** | Longer time from data collection to actionable insights. | Near-immediate insights from incoming data streams. |

## Contributing

Your contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This awesome list is under the [MIT License](LICENSE).
