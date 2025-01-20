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

## Vendors / Platforms

| Vendor | Description | Feature Store |
|--------|-------------|---------------|
| **[Tecton](https://www.tecton.ai/)** | Real-time Feature Platform offering Feature Engine, Feature Store, and Feature Repo for comprehensive Feature Management. Supports batch and streaming features, ensures data correctness, and provides low-latency serving for real-time ML applications. | Yes, integrated |
| **[Hazelcast](https://hazelcast.com/)** | Unified Real-Time Data Platform with distributed compute engine and fast data store for Stream Processing. Enables real-time feature computation and model serving for ML applications. | No, but integrates with Feast |
| **[Hopsworks](https://www.hopsworks.ai/)** | Serverless Feature Store supporting both online and offline features for real-time ML. Includes Model Management, Vector Database, and data-mesh architecture for efficient feature serving. | Yes, core offering |
| **[Iguazio](https://www.iguazio.com/)** | End-to-end platform focused on real-time ML and GenAI, offering Data Management, Development & CI/CD, Deployment, and Monitoring & LiveOps for ML models. | Yes, integrated |
| **[Xenonstack](https://www.xenonstack.com/)** | End-to-end real-time analytics platform, offering IoT Analytics, Stream and Batch Processing Integration, and Streaming Visualization Solutions for real-time ML applications. | Not specified |
| **[Abacus AI](https://abacus.ai/)** | Comprehensive platform offering LLMOps and MLOps solutions, including Custom Fine-Tunes, Vector Store, and AI Agent Workflow for real-time ML model deployment and serving. | Yes, Real-Time ML Feature Store |
| **[Rockset](https://rockset.com/)** | Real-time analytics database offering fast search, filtering, aggregations, and joins using standard SQL. Enables real-time feature computation and model serving for ML applications. | Not specified |
| **[Chalk AI](https://chalk.ai/)** | Data platform for real-time machine learning with feature pipelines, built-in scheduling, streaming, and caching for efficient feature computation and serving. | Yes, data source agnostic |
| **[Fennel](https://fennel.ai/)** | Real-Time Feature Platform for authoring, computing, storing, serving, monitoring, and governing real-time and batch ML features. | Yes, integrated |
| **[Featurebyte](https://www.featurebyte.com/)** | Feature Platform offering automatic feature generation, experimentation on live data, and AI data pipeline deployment for real-time ML applications. | Yes, integrated |
| **[Featureform](https://www.featureform.com/)** | Open Source Feature Store for defining, managing, and deploying ML features in real-time. | Yes, core offering |
| **[Molecula](https://www.molecula.com/)** | Data-centric AI infrastructure and database (Featurebase) with AI Automation and knowledge retrieval Platform (Ensemble) for real-time ML applications. | Yes, one of the first online feature stores |
| **[Vespa](https://vespa.ai/)** | Fully featured search engine and vector database with cloud offering, providing real-time feature-store functionality for ML applications. | Yes, real-time feature-store functionality |
| **[H2O.ai](https://h2o.ai/)** | Comprehensive AI platform offering solutions from MLOps to LLMOps and SLMs, including real-time feature management and automation for ML applications. | Yes, with feature management and automation |
| **[SingleStore](https://www.singlestore.com/)** | Real-time data platform for reading, writing, and reasoning on petabyte-scale data, enabling real-time feature computation and model serving for ML applications. | Can be used as a feature store |
| **[DataStax](https://www.datastax.com/)** | Suite of data management solutions built on Apache Cassandra for handling real-time data at scale, supporting real-time feature computation for ML models. | Can be used as a feature store |
| **[DataRobot](https://www.datarobot.com/)** | Complete ML and GenAI platform with Automated Machine Learning and Feature Engineering, supporting real-time feature computation and model serving. | Can integrate with external feature stores |
| **[Feast](https://feast.dev/)** | Open Source Feature Store, widely used and integrated with various ML projects and vendors for real-time feature serving. | Yes, core offering |
| **[Bytewax](https://bytewax.io/)** | Open-source stream processing framework for building real-time data applications in Python, enabling real-time feature computation for ML models. | Not specified |
| **[Kaskada](https://kaskada.com/)** | Real-Time AI streaming engine offering real-time aggregation, event detection, and history replay for ML feature computation. | Yes, integrated |

## Conferences

| Conference | Date | Location | Format |
|------------|------|----------|--------|
| **[Real-Time Analytics Summit 2025](https://startree.ai/rta-summit)** | May 14, 2025 | Virtual | Virtual |
| **[Feature Store Summit 2024](https://www.featurestoresummit.com/)** | October 15, 2024 | Virtual | Virtual |
| **[MLOps World: Annual Machine Learning in Production Summit](https://mlopsworld.com/)** | November 7-8, 2024 | Varies (past summits in San Francisco, Amsterdam, Tokyo) | Hybrid |
| **[Data + AI Summit 2025](https://www.databricks.com/dataaisummit)** | June 9-12, 2025 | San Francisco | Hybrid |
| **[Conference on Machine Learning and Systems (MLSys) 2025](https://mlsys.org/)** | May 12-15, 2025 | Santa Clara | In-person |
| **[AI & Big Data Expo North America 2025](https://www.ai-expo.net/northamerica/)** | June 4-5, 2025 | Santa Clara | In-person |
| **[Data Innovation Summit 2025](https://datainnovationsummit.com/)** | May 7-8, 2025 | Kistamässan, Stockholm | Hybrid |
| **[Machine Learning Prague 2025](https://www.mlprague.com/)** | April 28, 2025 | Prague | Hybrid |


## Contributing

Your contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This awesome list is under the [MIT License](LICENSE).
