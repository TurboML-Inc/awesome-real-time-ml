# Awesome Real-Time Machine Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg) [![X](https://img.shields.io/badge/X-%23000000?logo=X&logoColor=white)](https://twitter.com/turboml)


Here's a curated list of awesome real-time machine learning blogs, videos, tools and platforms, conferences, research papers, etc.

## Table of Contents
- [🤔 What even is "real-time" Machine Learning?](#what-even-is-real-time-machine-learning)
- [🆚 Traditional ML vs Real-Time ML](#traditional-ml-vs-real-time-ml)
- [🛠️ Tools & Workflow Stages](#tools--workflow-stages)
    - [📡 Event streaming platforms](#event-streaming-platforms)
    - [⚙️ Streaming Engines](#streaming-engines)
    - [🔧 Feature Engineering and Feature Stores](#feature-engineering-and-feature-stores)
    - [🧠 Model Development and Training](#model-development-and-training)
    - [📊 Experiment and Metadata Management](#experiment-and-metadata-management)
    - [🚀 Model Deployment and Serving](#model-deployment-and-serving)
    - [🔍 Monitoring and Feedback Loop](#monitoring-and-feedback-loop)
- [🏗️ Real-Time ML Internal Platform Resources](#real-time-ml-internal-platform-resources)
- [🎥 Videos](#videos)
- [🏢 Vendors / Platforms](#vendors--platforms)
- [🎪 Conferences](#conferences)

## What even is "real-time" Machine Learning?

Real-time Machine Learning (ML) delivers predictions and adapts models with extremely low latency, using fresh, continuously streaming data. It employs online or continual learning to instantly update models with new information, ensuring the most relevant insights for immediate actions. This dynamic approach contrasts with batch processing and is crucial for applications requiring instant responsiveness to changing patterns.

- **Real-Time Predictions**: Model outputs generated on-demand as data arrives with extremely low latency.
- **Real-Time Features**: Input attributes derived from real-time, rapidly changing data, processed quickly.
- **Real-Time Learning**: Continuous model updating (online or continual learning) using new data for adaptation and improvement of model performance over time.

## Traditional ML vs Real-Time ML

| Aspect | Traditional ML | Real-Time ML |
|--------|----------------|--------------|
| **Data Processing** | Processes static, historical datasets in batches. | Continuously ingests and processes streaming data in real-time. |
| **Model Training** | Models are trained offline using complete datasets. | Models are updated incrementally as new data arrives, often using online learning algorithms. |
| **Latency** | Can tolerate higher latency in processing and predictions. | Requires low-latency processing and near-instantaneous predictions. |
| **Scalability** | Typically scales vertically with more powerful hardware. Horizontal scaling is possible with distributed frameworks. | Often requires horizontal scalability to handle high-volume data streams. |
| **Infrastructure** | Can run on standard computing resources. | Often requires specialized streaming infrastructure like Apache Kafka or Apache Flink. |
| **Adaptability** | Models are less adaptive to changing patterns without manual retraining. | Models can adapt to concept drift and evolving patterns in real-time. |
| **Feature Engineering** | Features are often engineered manually and in advance. | Features may be generated on-the-fly or use automated feature extraction techniques. |
| **Model Deployment** | Models are deployed as static versions, updated periodically. | Models are continuously updated and deployed in a streaming fashion. |
| **Use Cases** | Effective for predictive analytics, segmentation, and batch or streaming data predictions. | Ideal for fraud detection, real-time bidding, and personalized recommendations. |
| **Data Volume** | Can work effectively with smaller datasets. | Typically requires larger volumes of data for accurate real-time predictions. |
| **Computational Resources** | Generally less computationally intensive. | Optimizes computational resource usage by processing data incrementally, reducing the need for reprocessing entire datasets, but may require consistent resource availability for real-time updates. |
| **Monitoring** | Periodic model performance checks are usually sufficient unless operating in dynamic environments. | Requires continuous monitoring of model performance and data quality. |
| **Feedback Loop** | Feedback is incorporated in batch updates. | Immediate feedback integration for rapid model adjustments. |
| **Complexity** | Generally simpler to implement and maintain. | More complex, requiring specialized knowledge in streaming architectures and online learning algorithms. |
| **Time-to-Insight** | Longer time from data collection to actionable insights. | Near-immediate insights from incoming data streams. |

## Tools & Workflow Stages

1. ### Event streaming platforms
   - [Apache Kafka](https://kafka.apache.org/) ![](https://img.shields.io/github/stars/apache/kafka.svg?style=social) 
   - [Redpanda](https://github.com/redpanda-data/redpanda) ![](https://img.shields.io/github/stars/redpanda-data/redpanda.svg?style=social) 
   - [Apache Flink](https://flink.apache.org/) ![](https://img.shields.io/github/stars/apache/flink.svg?style=social) 
   - [Apache Pulsar](https://pulsar.apache.org/) ![](https://img.shields.io/github/stars/apache/pulsar.svg?style=social) 
   - [Amazon Kinesis](https://aws.amazon.com/kinesis/)
   - [Apache RocketMQ](https://github.com/apache/rocketmq/) ![](https://img.shields.io/github/stars/apache/rocketmq.svg?style=social) 
   - [AutoMQ](https://github.com/AutoMQ/automq/) ![](https://img.shields.io/github/stars/AutoMQ/automq.svg?style=social) 
   - [Fluvio](https://github.com/infinyon/fluvio/) ![](https://img.shields.io/github/stars/infinyon/fluvio.svg?style=social) 
   - [Gazette](https://github.com/gazette/core/) ![](https://img.shields.io/github/stars/gazette/core.svg?style=social) 
   - [NATS](https://nats.io/) ![](https://img.shields.io/github/stars/nats-io/nats-server.svg?style=social) 
   - [Nsq](https://github.com/nsqio/nsq/) ![](https://img.shields.io/github/stars/nsqio/nsq.svg?style=social) 

2. ### Streaming Engines
   - [Apache Flink](https://flink.apache.org/) ![](https://img.shields.io/github/stars/apache/flink.svg?style=social) 
   - [Apache Kafka Streams](https://kafka.apache.org/documentation/streams/) ![](https://img.shields.io/github/stars/apache/kafka.svg?style=social) 
   - [Apache Samza](https://samza.apache.org/) ![](https://img.shields.io/github/stars/apache/samza.svg?style=social) 
   - [Apache Spark](https://spark.apache.org/) ![](https://img.shields.io/github/stars/apache/spark.svg?style=social) 
   - [Apache Storm](https://storm.apache.org/) ![](https://img.shields.io/github/stars/apache/storm.svg?style=social) 
   - [Arroyo](https://arroyo.dev/) ![](https://img.shields.io/github/stars/ArroyoSystems/arroyo.svg?style=social)
   - [Bytewax](https://github.com/bytewax/bytewax) ![](https://img.shields.io/github/stars/bytewax/bytewax.svg?style=social) 
   - [Faust](https://github.com/faust-streaming/faust/) ![](https://img.shields.io/github/stars/faust-streaming/faust.svg?style=social) 
   - [Feldera](https://www.feldera.com/) ![](https://img.shields.io/github/stars/feldera/feldera.svg?style=social)
   - [Mantis](https://netflix.github.io/mantis/) ![](https://img.shields.io/github/stars/netflix/mantis.svg?style=social)
   - [Materialize](https://materialize.com/) ![](https://img.shields.io/github/stars/MaterializeInc/materialize.svg?style=social)
   - [Numaflow](https://github.com/numaproj/numaflow) ![](https://img.shields.io/github/stars/numaproj/numaflow.svg?style=social) 
   - [Pathway](https://pathway.com/)
   - [Quix Streams](https://github.com/quixio/quix-streams) ![](https://img.shields.io/github/stars/quixio/quix-streams.svg?style=social) 
   - [Scramjet Transform Hub](https://github.com/scramjetorg/scramjet) ![](https://img.shields.io/github/stars/scramjetorg/scramjet.svg?style=social) 
   - [Timeplus Proton](https://timeplus.io/) ![](https://img.shields.io/github/stars/timeplus-io/proton.svg?style=social) 
   - [HStreamDB](https://hstream.io/) ![](https://img.shields.io/github/stars/hstreamdb/hstream.svg?style=social) 
   - [eKuiper](https://github.com/lf-edge/ekuiper) ![](https://img.shields.io/github/stars/lf-edge/ekuiper.svg?style=social) 
   - [Warpstream](https://warpstream.io/)
   - [WindFlow](https://github.com/ParaGroup/WindFlow) ![](https://img.shields.io/github/stars/ParaGroup/WindFlow.svg?style=social) 
   - [RisingWave](https://www.risingwave.dev/)

3. ### Feature Engineering and Feature Stores
   - [Volga](https://github.com/volga-project/volga) ![](https://img.shields.io/github/stars/volga-project/volga.svg?style=social) 
   - [OpenMLDB](https://github.com/4paradigm/OpenMLDB) ![](https://img.shields.io/github/stars/4paradigm/OpenMLDB.svg?style=social) 
   - [Feathr](https://github.com/feathr-ai/feathr) ![](https://img.shields.io/github/stars/feathr-ai/feathr.svg?style=social) 
   - [Chronon](https://github.com/airbnb/chronon) ![](https://img.shields.io/github/stars/airbnb/chronon.svg?style=social) 
   - [Feathub](https://github.com/alibaba/feathub) ![](https://img.shields.io/github/stars/alibaba/feathub.svg?style=social) 
   - [FeatureForm](https://github.com/featureform/featureform) ![](https://img.shields.io/github/stars/featureform/featureform.svg?style=social) 
   - [Hopsworks](https://github.com/logicalclocks/hopsworks) ![](https://img.shields.io/github/stars/logicalclocks/hopsworks.svg?style=social) 
   - [Kaskada](https://github.com/kaskada-ai/kaskada) ![](https://img.shields.io/github/stars/kaskada-ai/kaskada.svg?style=social) 
   - [Feast](https://feast.dev/) ![](https://img.shields.io/github/stars/feast-dev/feast.svg?style=social) 
   - [Caraml](https://github.com/caraml-dev/caraml-store) ![](https://img.shields.io/github/stars/caraml-dev/caraml-store.svg?style=social) 
   - [Butterfree](https://github.com/quintoandar/butterfree) ![](https://img.shields.io/github/stars/quintoandar/butterfree.svg?style=social) 
   - [Bytehub](https://github.com/bytehub-ai/bytehub) ![](https://img.shields.io/github/stars/bytehub-ai/bytehub.svg?style=social) 
   - [inlined.io](https://github.com/inlinedio/ikv-store) ![](https://img.shields.io/github/stars/inlinedio/ikv-store.svg?style=social) 
   - [Tecton](https://www.tecton.ai/)
   - [Fennel](https://www.fennel.ai/)
   - [Chalk](https://www.chalk.ai/)
   - [Zipline](https://www.zipline.ai/)
   - [H2O Feature Store](https://h2o.ai/platform/ai-cloud/make/feature-store/)
   - [Qwak](https://www.qwak.com/)
   - [Iguazio](https://www.iguazio.com/)
   - [Amazon SageMaker Feature Store](https://aws.amazon.com/sagemaker/feature-store/)
   - [Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore)
   - [Databricks Feature Store](https://www.databricks.com/product/feature-store)
   - [Snowflake Feature Store](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)
   - [Microsoft Azure Feature Store](https://aka.ms/featurestore-get-started)

4. ### Model Development and Training
   - [Adapters](https://github.com/Adapter-Hub/adapter-transformers) ![](https://img.shields.io/github/stars/Adapter-Hub/adapter-transformers.svg?style=social) 
   - [AutoTrain Advanced](https://huggingface.co/autotrain)
   - [Composer](https://github.com/mosaicml/composer) ![](https://img.shields.io/github/stars/mosaicml/composer.svg?style=social) 
   - [Flax](https://flax.readthedocs.io/)
   - [H2O-3](https://www.h2o.ai/products/h2o/)
   - [Jax](https://github.com/google/jax) ![](https://img.shields.io/github/stars/google/jax.svg?style=social) 
   - [PEFT](https://github.com/huggingface/peft) ![](https://img.shields.io/github/stars/huggingface/peft.svg?style=social) 
   - [PyTorch](https://pytorch.org/) ![](https://img.shields.io/github/stars/pytorch/pytorch.svg?style=social) 
   - [scikit-learn](https://scikit-learn.org/) ![](https://img.shields.io/github/stars/scikit-learn/scikit-learn.svg?style=social) 
   - [SetFit](https://github.com/huggingface/setfit) ![](https://img.shields.io/github/stars/huggingface/setfit.svg?style=social) 
   - [TensorFlow](https://www.tensorflow.org/) ![](https://img.shields.io/github/stars/tensorflow/tensorflow.svg?style=social) 
   - [torchkeras](https://github.com/lyhue1991/torchkeras) ![](https://img.shields.io/github/stars/lyhue1991/torchkeras.svg?style=social) 

5. ### Workflow Orchestration
   - [Airflow](https://airflow.apache.org/) ![](https://img.shields.io/github/stars/apache/airflow.svg?style=social) 
   - [Argo Workflows](https://argoproj.github.io/argo-workflows/) ![](https://img.shields.io/github/stars/argoproj/argo-workflows.svg?style=social) 
   - [Dagster](https://dagster.io/)
   - [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/overview/) ![](https://img.shields.io/github/stars/kubeflow/pipelines.svg?style=social) 
   - [Metaflow](https://metaflow.org/)
   - [Prefect](https://www.prefect.io/)
   - [ZenML](https://github.com/zenml-io/zenml)
   - [MLflow](https://mlflow.org/)
   - [Guild AI](https://guild.ai/)
   - [Polyaxon](https://polyaxon.com/)
   - [Dask](https://www.dask.org/)
   - [Luigi](https://luigi.readthedocs.io/)
   - [Snakemake](https://snakemake.readthedocs.io/)
   - [Ray](https://ray.io/)
   - [Flyte](https://flyte.org/)
   - [KubeFlow](https://www.kubeflow.org/) ![](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=social) 

6. ### Experiment and Metadata Management
   - [AI2 Tango](https://github.com/allenai/tango) ![](https://img.shields.io/github/stars/allenai/tango.svg?style=social) 
   - [Aim](https://github.com/aimhubio/aim) ![](https://img.shields.io/github/stars/aimhubio/aim.svg?style=social) 
   - [Catalyst](https://github.com/catalyst-team/catalyst) ![](https://img.shields.io/github/stars/catalyst-team/catalyst.svg?style=social) 
   - [ClearML](https://github.com/allegroai/clearml) ![](https://img.shields.io/github/stars/allegroai/clearml.svg?style=social) 
   - [CodaLab](https://github.com/codalab/codalab-worksheets) ![](https://img.shields.io/github/stars/codalab/codalab-worksheets.svg?style=social) 
   - [Deepkit](https://github.com/deepkit/deepkit-ml) ![](https://img.shields.io/github/stars/deepkit/deepkit-ml.svg?style=social) 
   - [Dolt](https://github.com/dolthub/dolt) ![](https://img.shields.io/github/stars/dolthub/dolt.svg?style=social) 
   - [DVC](https://github.com/iterative/dvc) ![](https://img.shields.io/github/stars/iterative/dvc.svg?style=social) 
   - [Flor](https://github.com/ucbrise/flor) ![](https://img.shields.io/github/stars/ucbrise/flor.svg?style=social) 
   - [Guild AI](https://github.com/guildai/guildai) ![](https://img.shields.io/github/stars/guildai/guildai.svg?style=social) 
   - [Hangar](https://github.com/tensorwerk/hangar-py) ![](https://img.shields.io/github/stars/tensorwerk/hangar-py.svg?style=social) 
   - [Keepsake](https://github.com/replicate/keepsake) ![](https://img.shields.io/github/stars/replicate/keepsake.svg?style=social) 
   - [KitOps](https://github.com/jozu-ai/kitops) ![](https://img.shields.io/github/stars/jozu-ai/kitops.svg?style=social) 
   - [lakeFS](https://github.com/treeverse/lakeFS) ![](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=social) 
   - [MLflow](https://github.com/mlflow/mlflow) ![](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=social) 
   - [ModelDB](https://github.com/VertaAI/modeldb) ![](https://img.shields.io/github/stars/VertaAI/modeldb.svg?style=social) 
   - [ModelStore](https://github.com/operatorai/modelstore) ![](https://img.shields.io/github/stars/operatorai/modelstore.svg?style=social) 
   - [Neptune](https://github.com/neptune-ai/neptune-client) ![](https://img.shields.io/github/stars/neptune-ai/neptune-client.svg?style=social) 
   - [ormb](https://github.com/kleveross/ormb) ![](https://img.shields.io/github/stars/kleveross/ormb.svg?style=social) 
   - [Polyaxon](https://github.com/polyaxon/polyaxon) ![](https://img.shields.io/github/stars/polyaxon/polyaxon.svg?style=social) 
   - [Quilt](https://github.com/quiltdata/quilt) ![](https://img.shields.io/github/stars/quiltdata/quilt.svg?style=social) 
   - [Sacred](https://github.com/IDSIA/sacred) ![](https://img.shields.io/github/stars/IDSIA/sacred.svg?style=social) 
   - [Studio](https://github.com/studioml/studio) ![](https://img.shields.io/github/stars/studioml/studio.svg?style=social) 
   - [TerminusDB](https://github.com/terminusdb/terminusdb) ![](https://img.shields.io/github/stars/terminusdb/terminusdb.svg?style=social) 
   - [Weights & Biases](https://github.com/wandb/wandb) ![](https://img.shields.io/github/stars/wandb/wandb.svg?style=social) 

7. ### Model Deployment and Serving
   - [AirLLM](https://github.com/lyogavin/airllm) ![](https://img.shields.io/github/stars/lyogavin/airllm.svg?style=social) 
   - [Apache PredictionIO](https://github.com/apache/predictionio) ![](https://img.shields.io/github/stars/apache/predictionio.svg?style=social) 
   - [Apache TVM](https://tvm.apache.org/) ![](https://img.shields.io/github/stars/apache/tvm/.svg?style=social) 
   - [BentoML](https://github.com/bentoml/BentoML) ![](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=social) 
   - [Cortex](https://github.com/cortexlabs/cortex) ![](https://img.shields.io/github/stars/cortexlabs/cortex.svg?style=social) 
   - [DeepDetect](https://github.com/jolibrain/deepdetect) ![](https://img.shields.io/github/stars/olibrain/deepdetect.svg?style=social) 
   - [Hydrosphere Serving](https://github.com/Hydrospheredata/hydro-serving) ![](https://img.shields.io/github/stars/Hydrospheredata/hydro-serving.svg?style=social) 
   - [Jina](https://github.com/jina-ai/jina) ![](https://img.shields.io/github/stars/jina-ai/jina.svg?style=social) 
   - [KServe](https://github.com/kserve/kserve) ![](https://img.shields.io/github/stars/kserve/kserve.svg?style=social) 
   - [MindsDB](https://github.com/mindsdb/mindsdb) ![](https://img.shields.io/github/stars/mindsdb/mindsdb.svg?style=social) 
   - [MLRun](https://github.com/mlrun/mlrun) ![](https://img.shields.io/github/stars/mlrun/mlrun.svg?style=social) 
   - [MLServer](https://github.com/SeldonIO/mlserver) ![](https://img.shields.io/github/stars/SeldonIO/mlserver.svg?style=social) 
   - [ONNX Runtime](https://onnxruntime.ai/) ![](https://img.shields.io/github/stars/microsoft/onnxruntime.svg?style=social) 
   - [Seldon Core](https://github.com/SeldonIO/seldon-core) ![](https://img.shields.io/github/stars/SeldonIO/seldon-core.svg?style=social) 
   - [SkyPilot](https://github.com/skypilot-org/skypilot) ![](https://img.shields.io/github/stars/skypilot-org/skypilot.svg?style=social) 
   - [SparseML](https://github.com/neuralmagic/sparseml) ![](https://img.shields.io/github/stars/neuralmagic/sparseml.svg?style=social) 
   - [TorchServe](https://github.com/pytorch/serve) ![](https://img.shields.io/github/stars/pytorch/serve.svg?style=social) 
   - [Tensorflow Serving](https://github.com/tensorflow/serving) ![](https://img.shields.io/github/stars/tensorflow/serving.svg?style=social) 
   - [Triton Inference Server](https://github.com/triton-inference-server/server) ![](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=social) 

8. ### Monitoring and Feedback Loop
   - [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) ![](https://img.shields.io/github/stars/tatsu-lab/alpaca_eval.svg?style=social) 
   - [ARES](https://github.com/stanford-futuredata/ARES) ![](https://img.shields.io/github/stars/stanford-futuredata/ARES.svg?style=social) 
   - [AutoML Benchmark](https://github.com/openml/automlbenchmark) ![](https://img.shields.io/github/stars/openml/automlbenchmark.svg?style=social) 
   - [continuous-eval](https://github.com/relari-ai/continuous-eval) ![](https://img.shields.io/github/stars/relari-ai/continuous-eval.svg?style=social) 
   - [Deepchecks](https://github.com/deepchecks/deepchecks) ![](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=social) 
   - [DeepEval](https://github.com/confident-ai/deepeval) ![](https://img.shields.io/github/stars/confident-ai/deepeval.svg?style=social) 
   - [EvalAI](https://github.com/Cloud-CV/EvalAI) ![](https://img.shields.io/github/stars/Cloud-CV/EvalAI.svg?style=social) 
   - [Evals](https://github.com/openai/evals) ![](https://img.shields.io/github/stars/openai/evals.svg?style=social) 
   - [Evaluate](https://github.com/huggingface/evaluate) ![](https://img.shields.io/github/stars/huggingface/evaluate.svg?style=social) 
   - [Evidently](https://github.com/evidentlyai/evidently) ![](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=social) 
   - [Grafana](https://grafana.com/) ![](https://img.shields.io/github/stars/grafana/grafana.svg?style=social) 
   - [Giskard](https://github.com/Giskard-AI/giskard) ![](https://img.shields.io/github/stars/Giskard-AI/giskard.svg?style=social) 
   - [Helicone](https://github.com/Helicone/helicone) ![](https://img.shields.io/github/stars/Helicone/helicone.svg?style=social) 
   - [HELM](https://github.com/stanford-crfm/helm) ![](https://img.shields.io/github/stars/stanford-crfm/helm.svg?style=social) 
   - [Langfuse](https://github.com/langfuse/langfuse) ![](https://img.shields.io/github/stars/langfuse/langfuse.svg?style=social) 
   - [LightEval](https://github.com/huggingface/lighteval) ![](https://img.shields.io/github/stars/huggingface/lighteval.svg?style=social) 
   - [MLPerf Inference](https://github.com/mlcommons/inference) ![](https://img.shields.io/github/stars/mlcommons/inference.svg?style=social) 
   - [MTEB](https://github.com/embeddings-benchmark/mteb) ![](https://img.shields.io/github/stars/embeddings-benchmark/mteb.svg?style=social) 
   - [NannyML](https://github.com/NannyML/nannyml) ![](https://img.shields.io/github/stars/NannyML/nannyml.svg?style=social) 
   - [OpenCompass](https://github.com/open-compass/OpenCompass) ![](https://img.shields.io/github/stars/open-compass/OpenCompass.svg?style=social) 
   - [Optimum-Benchmark](https://github.com/huggingface/optimum-benchmark) ![](https://img.shields.io/github/stars/huggingface/optimum-benchmark.svg?style=social) 
   - [Phoenix](https://github.com/Arize-ai/phoenix)  ![](https://img.shields.io/github/stars/Arize-ai/phoenix.svg?style=social) 
   - [Prometheus](https://prometheus.io/) ![](https://img.shields.io/github/stars/prometheus/prometheus.svg?style=social) 
   - [PromptBench](https://github.com/microsoft/promptbench)  ![](https://img.shields.io/github/stars/microsoft/promptbench.svg?style=social) 
   - [Ragas](https://github.com/explodinggradients/ragas) ![](https://img.shields.io/github/stars/explodinggradients/ragas.svg?style=social) 
   - [WhyLabs](https://whylabs.ai/) ![](https://img.shields.io/github/stars/whylabs/whylabs-oss.svg?style=social)

## Real-Time ML Internal Platform Resources

1. **Picnic**

   *Industry*: e-commerce and grocery retail

   - [Picnic's Lakeless Data Warehouse](https://blog.picnic.nl/picnics-lakeless-data-warehouse-8ec02801d50b)
     This article discusses Picnic's data architecture, including near-real-time data processing for analytics.

   - [Running demand forecasting machine learning models at scale](https://jobs.picnic.app/en/blogs/running-demand-forecasting-machine-learning-models-at-scale)
     This blog post discusses Picnic's implementation of deep learning models for demand forecasting, including real-time prediction challenges.

   - [The trade-off between efficiency and being on-time: Optimizing drop times using machine learning](https://blog.picnic.nl/the-trade-off-between-efficiency-and-being-on-time-optimizing-drop-times-using-machine-learning-d3f6fb1b0f31)
     This blog post describes Picnic's use of machine learning for real-time optimization of delivery times.

2. **Netflix**

   *Industry*: Media and Entertainment, Streaming Services

   - [Supporting Diverse ML Systems at Netflix](https://netflixtechblog.com/supporting-diverse-ml-systems-at-netflix-2d2e6b6d205d)
     This article discusses Netflix's Machine Learning Platform (MLP) and how it supports various ML systems, including real-time applications.

   - [Optimizing the Netflix Streaming Experience with Data Science](http://techblog.netflix.com/2014/06/optimizing-netflix-streaming-experience.html)
     This blog post explores how Netflix uses real-time and near real-time algorithms along with machine learning models to optimize streaming experiences.

   - [Netflix Recommendations: Beyond the 5 stars (Part 1)](http://techblog.netflix.com/2012/04/netflix-recommendations-beyond-5-stars.html)
     This article discusses Netflix's recommendation system, including real-time ranking and machine learning experimentation with online A/B testing.

   - [Machine Learning Platform - Netflix Research](https://research.netflix.com/research-area/machine-learning-platform)
     This research area page describes Netflix's machine learning infrastructure, which supports both real-time high-throughput, low-latency use cases and high-volume batch workflows.

   - [Evolution of ML Fact Store](https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762)
     This article discusses Axion, Netflix's fact store for ML, which is used for real-time feature logging and offline feature generation to remove training-serving skew.

   - [Scaling Media Machine Learning at Netflix](https://netflixtechblog.com/scaling-media-machine-learning-at-netflix-f19b400243)
     This post outlines Netflix's media machine learning infrastructure, including real-time serving and searching of media feature values using systems like Marken.

   - [InTune: Reinforcement Learning-based Data Pipeline Optimization for Deep Learning Recommendation Models](https://research.netflix.com/publication/intune-reinforcement-learning-based-data-pipeline-optimization-for-deep)
     This research publication discusses Netflix's use of reinforcement learning for optimizing real-time data ingestion in deep learning recommendation models.

3. **Uber**

   *Industry*: Transportation and Technology

   - [Real-time Data Infrastructure at Uber](https://arxiv.org/pdf/2104.0087.pdf)
     This paper describes Uber's real-time data infrastructure, which processes petabytes of data daily to support various use cases including customer incentives, fraud detection, and machine learning model predictions.

   - [Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
     This article introduces Michelangelo, Uber's ML platform that enables internal teams to build, deploy, and operate machine learning solutions at scale, including real-time prediction capabilities.

   - [Building Scalable Streaming Pipelines for Near Real-Time Features](https://www.uber.com/en-IN/blog/building-scalable-streaming-pipelines/)
     This blog post discusses how Uber leverages Apache Flink to build real-time streaming pipelines for generating data and features for machine learning models.

   - [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/)
     This article explains how Uber uses deep neural networks for global ETA prediction, including real-time traffic information and low-latency requirements.

   - [Palette Feature Store: Uber's Centralized Feature Management Platform](https://www.uber.com/en-IN/blog/palette-meta-store-journey/)
     This resource describes Uber's Palette Feature Store, a centralized database of features used across the company for various machine learning projects, supporting both batch and near real-time use cases.

   - [Project RADAR: Intelligent Early Fraud Detection](https://www.uber.com/en-GB/blog/project-radar-intelligent-early-fraud-detection/)
     RADAR: Uber's AI system for real-time fraud detection and mitigation using time series analysis and pattern mining.

   - [How Uber Optimizes the Timing of Push Notifications using ML and Linear Programming](https://www.uber.com/en-IN/blog/how-uber-optimizes-push-notifications-using-ml/)
     Uber's real-time ML system optimizes push notification timing using XGBoost and linear programming for personalized user engagement.

   - [Uber's Real-Time Document Check](https://www.uber.com/en-GB/blog/ubers-real-time-document-check/)
     Uber's real-time ML system for instant ID verification, using on-device image quality checks and server-side document processing.

   - [Personalized Marketing at Scale: Uber's Out-of-App Recommendation System](https://www.uber.com/en-GB/blog/personalized-marketing-at-scale/)
     Uber's real-time ML system for personalized out-of-app recommendations, using location prediction and multi-stage ranking for billions of messages.

   - [Stopping Uber Fraudsters Through Risk Challenges](https://www.uber.com/en-GB/blog/stopping-uber-fraudsters-through-risk-challenges/)
     Uber's real-time ML system implements risk challenges like penny drop verification to detect and mitigate payment fraud dynamically.

4. **TikTok**

   *Industry*: Social Media, Entertainment, and Technology

   - [Monolith: Real-Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663)
     This paper describes TikTok's real-time recommendation system, Monolith, which uses collisionless embedding tables and online learning to adapt quickly to changing user preferences.

   - [Build TikTok's Personalized Real-Time Recommendation System in Python](https://2024.pycon.de/program/DPGRGW/)
     This tutorial demonstrates how to build a simplified version of TikTok's recommendation system using Python, including a feature store, vector database, and model serving infrastructure.

   - [Real-time Data Processing at TikTok](https://www.techaheadcorp.com/blog/decoding-tiktok-system-design-architecture/)
     This article discusses TikTok's use of technologies like Apache Kafka for real-time data streaming, enabling immediate processing of user interactions and content.

   - [The AI Algorithm that got TikTok Users Hooked](https://www.argoid.ai/blog/the-ai-algorithm-that-got-tiktok-users-hooked)
     This blog post explains key components of TikTok's AI algorithm and recommender system, including its self-training AI engine, content tags, and user profiles.

5. **Meta**

   *Industry*: Technology, Social Media, and Artificial Intelligence

   - [Spiral: Self-tuning services via real-time machine learning](https://engineering.fb.com/2018/06/28/data-infrastructure/spiral-self-tuning-services-via-real-time-machine-learning/)
     This article introduces Spiral, Meta's system for self-tuning high-performance infrastructure services at scale, using techniques that leverage real-time machine learning.

   - [Scaling data ingestion for machine learning training at Meta](https://engineering.fb.com/2022/09/19/ml-applications/data-ingestion-machine-learning-training-meta/)
     This blog post discusses Meta's experience building data ingestion and last-mile data preprocessing pipelines responsible for feeding data into AI training models, including real-time and near real-time processing.

   - [Meta's approach to machine learning prediction robustness](https://engineering.fb.com/2024/07/10/data-infrastructure/machine-learning-ml-prediction-robustness-meta/)
     This article outlines Meta's systematic framework for building prediction robustness, including real-time monitoring and auto-mitigation toolsets for calibration robustness.

   - [Machine Learning Platform - Meta Research](https://ai.meta.com/research/areas/machine-learning-platform/)
     This research area page describes Meta's machine learning infrastructure, which supports both real-time high-throughput, low-latency use cases and high-volume batch workflows.

   - [Inside Meta's AI optimization platform for engineers across the company](https://ai.meta.com/blog/looper-meta-ai-optimization-platform-for-engineers/)
     This blog post introduces Looper, Meta's end-to-end AI platform for optimization, personalization, and feedback collection, supporting 700 AI models and generating 4 million AI outputs per second.

   - [New AI advancements drive Meta's ads system performance and efficiency](https://ai.meta.com/blog/ai-ads-performance-efficiency-meta-lattice/)
     This blog post discusses Meta Lattice, a new model architecture that improves ad performance through real-time intent capture and multi-distribution modeling with temporal awareness.

   - [AI debugging at Meta with HawkEye](https://engineering.fb.com/2023/12/19/data-infrastructure/hawkeye-ai-debugging-meta/)
     This article introduces HawkEye, Meta's toolkit for monitoring, observability, and debuggability of end-to-end machine learning workflows powering ML-based products.

   - [How machine learning powers Facebook's News Feed ranking algorithm](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
      Facebook's News Feed uses real-time machine learning to personalize content ranking for billions of users, processing thousands of signals to predict engagement and optimize user experience.
     
   - [Scaling the Instagram Explore recommendations system](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
      Instagram's Explore uses a multi-stage real-time ML system with Two Tower neural networks and caching to recommend relevant content from billions of options.

6. **Google**

   *Industry*: Technology, Internet Services, and Artificial Intelligence

   - [Real-time ML analysis with TensorFlow, BigQuery, and Redpanda](https://www.redpanda.com/blog/real-time-machine-learning-processing-tensorflow-bigquery)
     This tutorial demonstrates how to build a real-time machine learning analysis system for fraud detection using Google Cloud services.

   - [Real-time AI with Google Cloud Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai)
     This blog post introduces Streaming Ingestion for Vertex AI Matching Engine and Feature Store, enabling real-time updates and low-latency retrieval of data for ML models.

   - [Streaming analytics solutions | Google Cloud](https://cloud.google.com/solutions/stream-analytics)
     This page describes Google Cloud's streaming analytics solutions for ingesting, processing, and analyzing event streams in real-time.

   - [Introduction to Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/overview)
     This documentation explains how Vertex AI Feature Store provides a centralized repository for organizing, storing, and serving ML features in real-time.

   - [Real-time Data Infrastructure at Google](https://arxiv.org/pdf/2104.0087.pdf)
     This paper describes Google's real-time data infrastructure, which processes petabytes of data daily to support various use cases including customer incentives, fraud detection, and machine learning model predictions.

   - [How Google Search ranking works](https://searchengineland.com/how-google-search-ranking-works-445141)
      Google's search ranking system uses real-time machine learning components like NavBoost and Twiddlers to dynamically adjust search results based on user behavior and current events.

7. **Spotify**

   *Industry*: Music Streaming, Technology, and Entertainment

   - [Unleashing ML Innovation at Spotify with Ray](https://engineering.atspotify.com/2023/02/unleashing-ml-innovation-at-spotify-with-ray/)
     This article discusses how Spotify uses Ray to empower ML practitioners, support diverse ML systems, and accelerate the user journey for ML research and prototyping.

   - [Product Lessons from ML Home: Spotify's One-Stop Shop for Machine Learning](https://engineering.atspotify.com/2022/01/product-lessons-from-ml-home-spotifys-one-stop-shop-for-machine-learning/)
     This post details ML Home, Spotify's internal user interface for their Machine Learning Platform, which provides capabilities for tracking experiments, visualizing results, and monitoring deployed models.

   - [The Winding Road to Better Machine Learning Infrastructure Through Tensorflow Extended and Kubeflow](https://engineering.atspotify.com/2019/12/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/)
     This blog post outlines Spotify's journey in establishing building blocks for their platformized Machine Learning experience, leveraging TensorFlow Extended (TFX) and Kubeflow.

   - [Feature Stores at Spotify: Building & Scaling a Centralized Platform](https://www.tecton.ai/apply/session-video-archive/feature-stores-at-spotify-building-scaling-a-centralized-platform/)
     This talk discusses Spotify's approach to building a centralized ML Platform in a highly autonomous organization, focusing on their feature store strategy.

   - [Real-time Data Infrastructure at Spotify](https://arxiv.org/pdf/2104.0087.pdf)
     This paper describes Spotify's real-time data infrastructure, which processes petabytes of data daily to support various use cases including customer incentives, fraud detection, and machine learning model predictions.

   - [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part I)](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/)
      Spotify's journey in implementing real-time ML models for personalized content recommendations on their Home page, including challenges in evaluation and automated deployment.

   - [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part II)](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-ii/)
      Spotify's journey in implementing real-time ML models for personalized content recommendations on their Home page, including challenges in evaluation and automated deployment.

8. **Instacart**

   *Industry*: E-commerce, Grocery Delivery, Technology

   - [Lessons Learned: The Journey to Real-Time Machine Learning at Instacart](https://www.instacart.com/company/how-its-made/lessons-learned-the-journey-to-real-time-machine-learning-at-instacart/)
     This article discusses Instacart's transition from batch-oriented ML systems to real-time, including challenges faced and solutions implemented for real-time serving and features.

   - [How Instacart Modernized the Prediction of Real Time Availability for Hundreds of Millions of Items](https://tech.instacart.com/how-instacart-modernized-the-prediction-of-real-time-availability-for-hundreds-of-millions-of-items-59b2a82c89fe)
     This blog post details Instacart's new real-time item availability model, which combines general, trending, and real-time scores to improve prediction accuracy and reduce computation costs.

   - [Predicting the real-time availability of 200 million grocery items](https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe)
     This article explains how Instacart uses machine learning to predict real-time availability of hundreds of millions of grocery items across the US and Canada, including their optimized scoring pipeline.

   - [Griffin: How Instacart's ML Platform Tripled ML Applications in a year](https://www.instacart.com/company/how-its-made/griffin-how-instacarts-ml-platform-tripled-ml-applications-in-a-year/)
     This post introduces Instacart's MLOps platform, Griffin, which includes components for real-time recommendations and other ML applications.

   - [How Instacart Used Data Streaming to Meet COVID-19 Challenges](https://www.confluent.io/customers/instacart/)
     This case study describes how Instacart leveraged Confluent Cloud for data streaming to create real-time availability models and fraud detection systems.

   - [Distributed Machine Learning at Instacart](https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423)
     This article discusses Instacart's distributed ML architecture, including parallel fulfillment ML jobs for real-time applications like batching, routing, and ETA prediction.

   - [Instacart's Item Availability Architecture: Solving for scale and consistency](https://tech.instacart.com/instacarts-item-availability-architecture-solving-for-scale-and-consistency-f5661acb20a6)

   - [Supercharging ML/AI Foundations at Instacart](https://tech.instacart.com/supercharging-ml-ai-foundations-at-instacart-d48214a2b511)
     This blog post discusses Instacart's efforts to improve their ML infrastructure, including faster feature onboarding and retrieval, which are crucial for real-time ML applications.

   - [Real-time Fraud Detection with Yoda and ClickHouse](https://tech.instacart.com/real-time-fraud-detection-with-yoda-and-clickhouse-bd08e9dbe3f4)

9. **DoorDash**

   *Industry*: Food Delivery, Technology, and Logistics

   - [Building Scalable Real-Time Event Processing with Kafka and Flink](https://doordash.engineering/2022/08/02/building-scalable-real-time-event-processing-with-kafka-and-flink/)
     This article discusses DoorDash's migration to a cloud-native streaming platform powered by Apache Kafka and Apache Flink for continuous stream processing and data ingestion into Snowflake.

   - [How DoorDash Built an Ensemble Learning Model for Time Series Forecasting](https://careersatdoordash.com/blog/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/)
     This blog post details DoorDash's implementation of ELITE, an ensemble learning model for efficient and accurate time series forecasting, used for weekly order volume predictions and delivery time forecasts.

   - [Building a Gigascale ML Feature Store with Redis](https://careersatdoordash.com/blog/building-a-gigascale-ml-feature-store-with-redis/)
     This article describes how DoorDash optimized their Redis-based feature store to handle tens of millions of reads per second, enabling real-time machine learning predictions at scale.

   - [Using ML and Optimization to Solve DoorDash's Dispatch Problem](https://careersatdoordash.com/blog/using-ml-and-optimization-to-solve-doordashs-dispatch-problem/)
     This blog post explains how DoorDash uses machine learning and optimization techniques to solve the complex dispatch problem of efficiently matching orders with drivers.

   - [Maintaining Machine Learning Model Accuracy Through Monitoring](https://careersatdoordash.com/blog/monitor-machine-learning-model-drift/)
     This article discusses DoorDash's approach to monitoring machine learning models in production, ensuring their continued accuracy and performance.

   - [Building Riviera: A Declarative Real-Time Feature Engineering Framework](https://careersatdoordash.com/blog/building-a-declarative-real-time-feature-engineering-framework/)
     This post describes DoorDash's development of Riviera, a framework for real-time feature engineering that allows data scientists to specify feature computation logic and production requirements through high-level constructs.

   - [Engineering Systems for Real-Time Predictions @DoorDash](https://www.infoq.com/presentations/doordash-real-time-predictions/)
     This presentation by Raghav Ramesh discusses DoorDash's approach to structuring machine learning systems in production for robust and wide-scale deployment of real-time predictions.

   - [How DoorDash Upgraded a Heuristic with ML to Save Thousands of Canceled Orders](https://careersatdoordash.com/blog/how-doordash-upgraded-a-heuristic-with-ml-to-save-thousands-of-canceled-orders/)
      DoorDash implemented real-time ML to replace heuristics, reducing order cancellations by predicting and mitigating potential issues.

   - [Personalizing the DoorDash Retail Store Page Experience](https://careersatdoordash.com/blog/personalizing-the-doordash-retail-store-page-experience/)
      DoorDash uses real-time ML to personalize retail store pages, dynamically adjusting content based on user preferences and behavior.

   - [Homepage Recommendation with Exploitation and Exploration](https://careersatdoordash.com/blog/homepage-recommendation-with-exploitation-and-exploration/)
      DoorDash's homepage employs real-time ML for personalized recommendations, balancing exploitation of known preferences with exploration of new options.

   - [Managing Supply and Demand Balance Through Machine Learning](https://careersatdoordash.com/blog/managing-supply-and-demand-balance-through-machine-learning/)
      DoorDash utilizes real-time ML models to dynamically balance supply and demand, optimizing resource allocation across their platform.

   - [3 Changes to Expand DoorDash's Product Search Beyond Delivery](https://careersatdoordash.com/blog/3-changes-to-expand-doordashs-product-search/)
      DoorDash enhanced their product search with real-time ML, improving relevance and expanding beyond delivery to include pickup and other services.

   - [Improving ETAs with multi-task models, deep learning, and probabilistic forecasts](https://careersatdoordash.com/blog/improving-etas-with-multi-task-models-deep-learning-and-probabilistic-forecasts/)
      DoorDash leverages real-time ML with multi-task models and deep learning to provide more accurate and probabilistic delivery time estimates.

   - [Beyond the Click: Elevating DoorDash's personalized notification experience with GNN recommendation](https://careersatdoordash.com/blog/doordash-customize-notifications-how-gnn-work/)
      DoorDash implements real-time Graph Neural Networks (GNN) to personalize notifications, enhancing user engagement beyond simple click-based recommendations.

10. **Booking.com**

    *Industry*: Travel and Technology

    - [Booking.com Launches New AI Trip Planner to Enhance Travel Planning Experience](https://news.booking.com/bookingcom-launches-new-ai-trip-planner-to-enhance-travel-planning-experience/)
      This announcement introduces Booking.com's AI Trip Planner, which uses machine learning models and large language model technology to create a conversational experience for trip planning.

    - [Booking.com Enhances Travel Planning with New AI-Powered Features for Easier, Smarter Decisions](https://news.booking.com/bookingcom-enhances-travel-planning-with-new-ai-powered-features--for-easier-smarter-decisions/)
      This article discusses Booking.com's expansion of AI-powered features, including Smart Filter, Property Q&A, and Review Summaries, which use Generative AI to simplify key steps in the trip planning process.

    - [Machine Learning in production: the Booking.com approach](https://booking.ai/https-booking-ai-machine-learning-production-3ee8fe943c70)
      This article discusses how Booking.com integrates machine learning into every step of the customer journey, detailing their approach to productionizing ML models.

    - [Booking.com: Building a Scalable Machine Learning Platform](https://h2o.ai/case-studies/booking-com/)
      This case study describes how Booking.com built a machine learning platform that scales to support their 200 data scientists and processes 1.5 million nights reserved every day.

    - [Booking.com accelerates data analysis with Confluent Cloud](https://www.confluent.io/customers/booking-com/)
      This customer story details how Booking.com transitioned from a self-managed Apache Kafka deployment to Confluent Cloud to improve reliability and enhance data management capabilities.

    - [Leverage graph technology for real-time Fraud Detection and Prevention](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5)

11. **Grab**

    *Industry*: Technology, Ride-Hailing, Food Delivery, and Digital Payments

    - [Real-time data ingestion in Grab](https://engineering.grab.com/real-time-data-ingestion)
      This article discusses Grab's approach to real-time data ingestion, which enables faster business decisions, optimizes data pipelines, and provides audit trails for fraud detection.

    - [How we got more accurate estimated time-of-arrivals in the app while pushing down tech costs](https://www.grab.com/sg/inside-grab/how-we-got-more-accurate-estimated-time-of-arrivals-in-the-app-while-pushing-down-tech-costs/)
      This blog post details how Grab consolidated their machine learning models for ETA prediction using their internal ML platform Catwalk, improving accuracy and reducing computing costs.

    - [GrabRideGuide, our new AI tool that predicts ride demand hotspots](https://www.grab.com/sg/inside-grab/stories/grabrideguide-our-new-ai-tool-that-predicts-high-demand-areas/)
      This article introduces GrabRideGuide, a fully automated real-time ride demand finder for Grab driver-partners, which uses AI to analyze past trends and suggest optimal routes.

    - [Evolution of Catwalk: Model serving platform at Grab](https://engineering.grab.com/catwalk-evolution)
      This post discusses Catwalk, Grab's ML model serving platform that powers their real-time decision-making capabilities across various services.

    - [Grab App at Scale with ScyllaDB](https://www.scylladb.com/2021/06/23/grab-app-at-scale-with-scylla/)
      This article describes how Grab uses ScyllaDB for real-time counters to detect fraud, identity, and safety risks, processing billions of events per day.

    - [Harnessing AI for public good: Grab's approach to AI Governance](https://www.grab.com/sg/inside-grab/stories/harnessing-ai-for-public-good-grabs-approach-to-ai-governance/)
      This blog post outlines Grab's use of AI and machine learning models for real-time automated decision-making to enhance customer experiences.

    - [Unsupervised graph anomaly detection - Catching new fraudulent behaviours](https://engineering.grab.com/graph-anomaly-model)

12. **Didact AI**

    *Industry*: Finance, Machine Learning, Stock Trading

    - [Didact AI: The anatomy of an ML-powered stock picking engine](https://principiamundi.com/posts/didact-anatomy/)
      This blog post details the architecture and technology behind Didact AI's machine learning-based stock picking engine, which consistently beat the S&P 500 for over a year on a weekly basis.

13. **Glassdoor**

    *Industry*: Technology, Job Search and Company Reviews

    - [Glassdoor uses machine learning to tell users if they're being paid fairly](https://www.zdnet.com/article/glassdoor-uses-machine-learning-to-tell-users-if-theyre-being-paid-fairly/)
      This article discusses how Glassdoor uses machine learning to analyze millions of salary reports and real-time supply and demand trends in local job markets to determine fair pay.

14. **Dailymotion**

    *Industry*: Video Sharing and Streaming, AdTech

    - [Case study Dailymotion - Martech Compass](https://martechcompass.com/case-study-dailymotion/)
      This case study discusses how Dailymotion implemented AI-based personalization to improve its video recommendation system, resulting in a 300% increase in unique video views.

    - [AI Video Intelligence: Innovation for Publishers & Broadcasters](https://pro.dailymotion.com/en/blog/ai-video-intelligence/)
      This blog post introduces Dailymotion's AI video intelligence technology, which analyzes video content to extract insights like themes, sentiment, and contextual markers for improved content tagging and contextual advertising.

    - [Dailymotion's Journey to Crafting the Ultimate Content-Driven Video Recommendation Engine with Qdrant Vector Database](https://qdrant.tech/blog/case-study-dailymotion/)
      This article details Dailymotion's implementation of a content-based recommendation system using Qdrant vector database, processing 420 million+ videos and serving 13 million+ recommendations daily.

    - [See Real time activity - Dailymotion Help Center](https://faq.dailymotion.com/hc/en-us/articles/360016890140-See-Real-time-activity)
      This help article explains Dailymotion's Real-time dashboard feature, which allows content creators to analyze views generated on their content or player embeds within the last 60 minutes and 24 hours.

    - [How Dailymotion reinvented itself to create a better user experience](https://www.builtinnyc.com/articles/spotlight-working-at-dailymotion-tech-innovation)
      This article discusses Dailymotion's transformation in 2017, including the development of algorithms to deliver personalized content and the integration of video player technology with a robust adtech platform.

15. **Coupang**

    *Industry*: E-commerce, Technology, and Logistics

    - [A symphony of people and technology: An inside look at a Coupang fulfillment center](https://www.aboutcoupang.com/English/news/news-details/2022/A-symphony-of-people-and-technology-An-inside-look-inside-a-Coupang-fulfillment-center/default.aspx)
      This article details how Coupang uses AI, machine learning, and real-time data in their fulfillment centers to optimize operations and improve employee experiences.

    - [How Coupang Conquered South Korean E-commerce](https://quartr.com/insights/company-research/how-coupang-conquered-south-korean-ecommerce)
      This article explains how Coupang uses AI to coordinate tasks for workers and drivers using real-time data, allocating labor and providing optimal routes.

    - [Matching duplicate items to improve catalog quality](https://medium.com/coupang-engineering/matching-duplicate-items-to-improve-catalog-quality-ca4abc827f94)
      Coupang utilizes real-time deep embedding vectors and FAISS for efficient similarity search to detect duplicate items across millions of products.

    - [Overcoming food delivery challenges with data science](https://medium.com/coupang-engineering/overcoming-food-delivery-challenges-with-data-science-6420cac1d59)
      Coupang Eats employs real-time machine learning for order assignment, dynamic pricing, and ETA predictions in food delivery.

16. **Slack**

    *Industry*: Technology, Communication, and Collaboration Software

    - [How Slack sends Millions of Messages in Real Time](https://blog.quastor.org/p/slack-sends-millions-messages-real-time)
      This blog post details Slack's architecture for sending millions of real-time messages daily across the globe, including their use of Channel Servers, Gateway Servers, and Presence Servers.

    - [Slack delivers native and secure generative AI powered by Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/slack-delivers-native-and-secure-generative-ai-powered-by-amazon-sagemaker-jumpstart/)
      This article discusses how Slack implemented generative AI capabilities using Amazon SageMaker JumpStart, ensuring data security and privacy for their customers.

    - [Real-Time Messaging Architecture at Slack](https://www.infoq.com/news/2023/04/real-time-messaging-slack/)
      This article provides insights into Slack's Pub/Sub architecture designed to manage real-time messages at scale, highlighting the challenges of delivering messages across different time zones and regions.

    - [How We Built Slack AI To Be Secure and Private](https://slack.engineering/how-we-built-slack-ai-to-be-secure-and-private/)
      This blog post explains Slack's approach to building AI features while maintaining rigorous standards for customer data stewardship, including their principles for secure and private AI implementation.

    - [Real Time Messaging API | Node Slack SDK](https://tools.slack.dev/node-slack-sdk/rtm-api/)
      This documentation describes Slack's Real Time Messaging API, which allows developers to receive events and send simple messages to Slack in real-time using WebSocket connections.

    - [Privacy principles: search, learning and artificial intelligence](https://slack.com/intl/en-in/trust/data-management/privacy-principles)
      This resource outlines Slack's privacy principles for their AI and machine learning implementations, emphasizing their commitment to data privacy and security.

    - [Email Classification](https://slack.engineering/email-classification/)
      Slack uses real-time machine learning to classify incoming email addresses for optimal collaboration suggestions.

17. **Swiggy**

    *Industry*: Food Delivery, Technology, and E-commerce

    - [Building Rock-Solid ML Systems](https://bytes.swiggy.com/building-rock-solid-ml-systems-d1d5ab544f0e)
      This blog post explores how Swiggy ensures ML reliability at scale by focusing on best practices that deliver consistent performance across their systems.

    - [Swiggy's Generative AI Journey: A Peek Into the Future](https://bytes.swiggy.com/swiggys-generative-ai-journey-a-peek-into-the-future-2193c7166d9a)
      This article discusses Swiggy's implementation of AI-powered neural search to help users discover food and groceries in a conversational manner, receiving tailored recommendations.

    - [Swiggylytics: Swiggy's real-time Analytics SDK](https://bytes.swiggy.com/swiggylytics-5046978965dc)
      This blog post details Swiggy's customizable analytics SDK for real-time data, enabling remote configuration and marking of real-time events.

    - [Hyperlocal Forecasting at Scale: The Swiggy Forecasting platform](https://bytes.swiggy.com/hyperlocal-forecasting-at-scale-the-swiggy-forecasting-platform-c07ecd5f5b86)
      This article discusses Swiggy's centralized forecasting service, which enables end-users to generate accurate forecasts quickly and cost-effectively.

    - [Enabling data science at scale at Swiggy: the DSP story](https://bytes.swiggy.com/enabling-data-science-at-scale-at-swiggy-the-dsp-story-208c2d85faf9)
      This blog post introduces Swiggy's Data Science Platform (DSP), an in-house ML deployment and orchestration platform that supports hundreds of real-time and batch models generating over 1 billion predictions per day.

    - [Deploying deep learning models at scale at Swiggy: Tensorflow Serving on DSP](https://bytes.swiggy.com/deploying-deep-learning-models-at-scale-at-swiggy-tensorflow-serving-on-dsp-ad5da40f7a6c)
      This article details Swiggy's implementation of Tensorflow Serving capability on their Data Science Platform for deploying deep learning models at scale.

    - [An ML approach for routing payment transactions](https://bytes.swiggy.com/an-ml-approach-for-routing-payment-transactions-5a14efb643a8)
      This blog post explains how Swiggy uses machine learning to optimally route payment transactions to different payment gateways, improving payment success rates.

    - [How ML Powers — When is my order coming?](https://bytes.swiggy.com/how-ml-powers-when-is-my-order-coming-part-ii-eae83575e3a9)
      Swiggy uses real-time machine learning models to provide accurate delivery time estimates, considering various dynamic factors affecting order fulfillment.

18. **Nubank**

    *Industry*: Financial Technology (Fintech), Digital Banking

    - [Presenting Precog, Nubank's Real Time Event AI](https://building.nubank.com.br/presenting-precog-nubanks-real-time-event-ai/)
      This article introduces Precog, Nubank's AI solution designed to improve customer service by efficiently routing calls using customer embeddings and features, significantly enhancing the customer experience.

    - [Real-time machine learning models in real life](https://building.nubank.com.br/real-time-machine-learning-models-in-real-life/)
      This blog post details Nubank's approach to implementing real-time machine learning models, discussing challenges and solutions for fast inference and real-time pipelines.

    - [Fklearn: Nubank's machine learning library](https://building.nubank.com.br/introducing-fklearn-nubanks-machine-learning-library-part-i-2/)
      This article introduces Fklearn, Nubank's in-house machine learning library that powers a broad set of ML models, solving problems from credit scoring to automated customer support chat responses.

    - [The potential of sequential modeling in fraud prevention](https://building.nubank.com.br/the-potential-of-sequential-modeling-in-fraud-prevention-insights-from-experts-at-nubank/)
      This post discusses how Nubank leverages sequential modeling, an advanced machine learning technique, to detect and prevent fraud in real-time.

    - [Nubank acquires Hyperplane to accelerate AI-first strategy](https://www.fintechfutures.com/2024/06/brazils-nubank-buys-ai-powered-data-intelligence-start-up-hyperplane/)
      This article discusses Nubank's acquisition of Hyperplane, a data intelligence company, to enhance its AI capabilities for providing more personalized financial products and services.

    - [Beyond prediction machines](https://building.nubank.com.br/beyond-prediction-machines/)
      Nubank explores causal inference techniques beyond traditional ML for real-time decision-making in credit limits, interest rates, and marketing strategies.
      

## Videos

1. **"Bring the power of machine learning to the world of streaming data"**  
   This video from Google Cloud Next demonstrates how to deploy and manage complete ML pipelines for real-time inference and predictions using Dataflow ML.  
   [Watch here](https://www.youtube.com/watch?v=SGsD0K1s8cI)

2. **"Jukebox: Spotify's Feature Infrastructure"**  
   Explains how Spotify manages features for machine learning, including their approach to feature stores and real-time feature serving.  
   [Watch here](https://www.youtube.com/watch?v=qv2DtDvpWx8)

3. **"Scaling Up Machine Learning in Instacart Search for the 2020 Surge"**  
   Discusses how Instacart scaled up its machine learning capabilities to handle the surge in demand during 2020, likely including real-time aspects of their search system.  
   [Watch here](https://www.youtube.com/watch?v=qRBFM3iGyhA)

4. **"How Booking.com Used Data Streaming to Put Travel Decisions into Customers' Hands"**  
   Explains how Booking.com leveraged data streaming to provide a comprehensive booking experience, including the use of Confluent's data streaming platform.  
   [Watch here](https://www.youtube.com/watch?v=1Tg_MEwSyRc)

5. **"Inside Coupang's AI-Powered Fulfillment Center"**  
   Showcases Coupang's newest fulfillment center, highlighting its AI-directed nerve center and army of robots for efficient operations.  
   [Watch here](https://www.youtube.com/watch?v=ZboD8Rg4j4k)

6. **"Real-time Machine Learning: Architecture and Challenges"**  
   Explores architectures and challenges in implementing real-time machine learning systems, emphasizing the importance of fresh data and low-latency predictions.  
   [Watch here](https://www.youtube.com/watch?v=S4A8QWN1G7s)

7. **"Batch-scoring vs Real-time ML systems"**  
   Compares batch scoring and real-time machine learning systems, discussing their advantages, disadvantages, and implementation differences.  
   [Watch here](https://www.youtube.com/watch?v=sVodCJyo6I8)

8. **"Journey to Real-Time ML: A Look at Feature Platforms & Modern RT ML Architectures Using Tecton"**  
   Demonstrates how to build a robust MLOps platform using MLflow and Tecton on Databricks for managing real-time ML models and features, with insights from FanDuel's implementation.  
   [Watch here](https://www.youtube.com/watch?v=j9rolKGk1Ns)

9. **"How to Build a Real Time ML Pipeline for Fraud Prediction"**  
   Demonstrates how to build a machine learning pipeline with real-time feature engineering for fraud detection, using Iguazio's data science platform to streamline the process from data ingestion to model deployment and monitoring.  
   [Watch here](https://www.youtube.com/watch?v=55Bsm5x1WzM)

10. **"Real-Time ML: Features and Inference // Sasha Ovsankin and Rupesh Gupta // MLOps Podcast #135”**  
    Explores challenges and solutions in implementing real-time machine learning features and inference at LinkedIn.  
    [Watch here](https://www.youtube.com/watch?v=cZpGgobIFxU)

11. **"Need for Speed: Machine Learning in the Era of Real-Time"**  
    Explores the evolution of real-time machine learning, discussing challenges in latency, data freshness, and resource efficiency, while providing insights on implementing RTML solutions.  
    [Watch here](https://www.youtube.com/watch?v=MWagqntxgeg)

12. **"Real-Time Event Processing for AI/ML with Numaflow // Sri Harsha Yayi // DE4AI"**  
    Discusses Intuit's development of Numaflow, an open-source platform designed to simplify event processing and inference on streaming data for machine learning applications.  
    [Watch here](https://www.youtube.com/watch?v=-EAexlXbN1I)

13. **"ML Batch vs streaming vs real-time data processing"**  
    Compares batch, streaming, and real-time data processing for machine learning, discussing misconceptions, costs, and decision-making criteria.  
    [Watch here](https://www.youtube.com/watch?v=87lj-A5vvAQ)

14. **"Machine Learning is Going Real-Time"**  
    Chip Huyen explores the state, use cases, solutions, and challenges of real-time machine learning in production across US and Chinese companies.  
    [Watch here](https://www.youtube.com/watch?v=t0gcsYEnHRY)

15. **"Realtime Stock Market Anomaly Detection using ML Models | An End to End Data Engineering Project"**  
    Demonstrates building a real-time anomaly detection system for stock market data using Quix Streams, Redpanda, and Docker.  
    [Watch here](https://www.youtube.com/watch?v=RUfVVOhihEA)

16. **"Real Time ML: Challenges and Solutions - Chip Huyen"**  
    Explores challenges in implementing real-time machine learning systems, including latency, train-predict inconsistency, and managing streaming infrastructure, while discussing potential solutions and architectures.  
    [Watch here](https://www.youtube.com/watch?v=kbxF3fT0YAI)

17. **"Real-time ML Model Monitoring with Data Sketches and Apache Pinot"**  
    Demonstrates how Uber leverages Apache Pinot as a data sketch store for ML model monitoring, using data profiling and sketch-based solutions to enable efficient and scalable monitoring across different data sources.  
    [Watch here](https://www.youtube.com/watch?v=sDkZqZbWNq8)

18. **"MLOps vs ML Orchestration // Ketan Umare // MLOps Podcast #183"**  
    Explores real-time machine learning challenges in traffic prediction and fraud detection, highlighting the importance of buffering and damping reactions in ML systems.  
    [Watch here](https://www.youtube.com/watch?v=vAt-FGqHSn0)

19. **"Lessons Learned: The Journey to Real-Time Machine Learning at Instacart"**  
    Guanghua Shu discusses Instacart's transition from batch-oriented to real-time ML systems, covering infrastructure changes, use cases, and key lessons learned in implementing real-time ML for their e-commerce platform.  
    [Watch here](https://www.youtube.com/watch?v=a9aoRDL4gJ0)

20. **"Leveraging GraphQL for Continual Learning in Real-Time ML Systems"**  
    Discusses how to set up real-time infrastructure and continual learning using GraphQL for machine learning systems, addressing limitations of batch-training paradigms and enabling adaptive models.  
    [Watch here](https://www.youtube.com/watch?v=21pJJ4J86zA)

21. **"Real-Time ML Insights with Richmond Alake"**  
    Explores real-time machine learning tools, techniques, and career insights with Richmond Alake, a Machine Learning Architect at Slalom Build, covering his work experiences and AI startups.  
    [Watch here](https://www.youtube.com/watch?v=OCHahSA1-YM)

22. **"Why real time event streaming pattern is indispensable for an AI native future"**  
    Discusses the importance of distributed event streaming for real-time analytics and AI-powered experiences, exploring its applications in data collection, enrichment, and measuring drift and explainability.  
    [Watch here](https://www.youtube.com/watch?v=zCey3ZXpEZk)

23. **"Realtime Prediction with Machine Learning and Data Transform with Redpanda"**  
    Explores building efficient AI applications using stateless pipelines and WebAssembly-powered streaming data transforms, demonstrating how to simplify data architecture for real-time analytics and machine learning.  
    [Watch here](https://www.youtube.com/watch?v=wOJnqdjeaEc)

24. **"Build Real-time Machine Learning Apps on Generative AI with Kafka Streams"**  
    Stepan Hinger discusses integrating large language models with data streaming using Kafka, demonstrating real-time AI applications for unstructured data analysis, customer support automation, and business intelligence through a framework called Area.  
    [Watch here](https://www.youtube.com/watch?v=sgWxkFV7U0g)

25. **"Feeding ML models with the data from the databases in real-time - DevConf.CZ 2024"**  
    Vojtech Juranek demonstrates how to use Debezium to ingest real-time data from databases into machine learning models, showcasing a live demo with TensorFlow and discussing challenges and solutions in implementing such systems.  
    [Watch here](https://www.youtube.com/watch?v=rem9QFHlHIU)

26. **"Architecting Data and Machine Learning Platforms"**  
    Marco Tranquillin and Firat Tekiner preview their upcoming book, covering the entire data lifecycle in cloud environments, from ingestion to activation, with a cloud-agnostic approach to data and ML platform architecture.  
    [Watch here](https://www.youtube.com/watch?v=uVifHHNpheA)

27. **"Real-Time ML Workflows at Capital One with Disha Singla"**  
    Disha Singla, Senior Director of Machine Learning Engineering at Capital One, discusses democratizing AI through reusable libraries and workflows for citizen data scientists, focusing on time series analysis, anomaly detection, and fraud prevention.  
    [Watch here](https://www.youtube.com/watch?v=2ZD8pwLMOtM)

28. **"ML Auto-Retraining: Update Your Model in Real Time"**  
    Discusses real-time ML techniques for cybersecurity, including aggregate engines for adapting to attacker shifts, Redis-based key-value stores for tracking indicators of compromise, and an auto-retraining framework for regularly updating models on different cadences.  
    [Watch here](https://www.youtube.com/watch?v=vt8FE7YYudY)

29. **"Real-Time Data Processing for ML Feature Engineering | Weiran Liu and Ping Chen"**  
    Discusses Meta's evolution of real-time data processing infrastructure for machine learning, covering applications in recommendation systems, content understanding, and fraud detection, with a focus on their latest platform "Extreme" and its use in real-time feature engineering.  
    [Watch here](https://www.youtube.com/watch?v=ga2StG7DjF4)

30. **"Building Real-Time ML Pipelines: Challenges and Solutions"**  
    Yaron Haviv discusses challenges in productizing AI/ML, introduces MLOps and feature stores, and demonstrates building real-time ML pipelines using the open-source MLRun framework, with examples of churn prediction and fraud detection use cases.  
    [Watch here](https://www.youtube.com/watch?v=ms2OU3noTOo)

31. **"Apache Spark and Apache Kafka for Real-Time Machine Learning"**  
    This webinar explores the integration of Apache Kafka and Apache Spark for building scalable real-time machine learning pipelines, covering fundamentals of real-time ML, challenges faced by data teams, and optimal usage of these technologies for data processing and analysis.  
    [Watch here](https://dzone.com/events/video-library/apache-spark-and-apache-kafka=for-real-time-machine=learning)


## Vendors / Platforms

| Vendor | Description | Feature Store |
|--------|-------------|---------------|
| **[Tecton](https://www.tecton.ai/)** | Real-time Feature Platform offering Feature Engine, Feature Store, and Feature Repo for comprehensive Feature Management. Supports batch and streaming features, ensures data correctness, and provides low-latency serving for real-time ML applications. | Yes, integrated |
| **[Hazelcast](https://hazelcast.com/)** | Unified Real-Time Data Platform with distributed compute engine and fast data store for Stream Processing. Enables real-time feature computation and model serving for ML applications. | Can integrate with external feature stores |
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
| **[Bytewax](https://bytewax.io/)** | Open-source stream processing framework for building real-time data applications in Python, enabling real-time feature computation for ML models. | Can integrate with external feature stores |
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
