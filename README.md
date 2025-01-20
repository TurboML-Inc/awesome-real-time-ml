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

## Real-Time ML Internal Platform Resources

1. **Picnic** (e-commerce and grocery retail)

- [Picnic's Lakeless Data Warehouse](https://blog.picnic.nl/picnics-lakeless-data-warehouse-8ec02801d50b)
- [Running demand forecasting machine learning models at scale](https://jobs.picnic.app/en/blogs/running-demand-forecasting-machine-learning-models-at-scale)
- [The trade-off between efficiency and being on-time: Optimizing drop times using machine learning](https://blog.picnic.nl/the-trade-off-between-efficiency-and-being-on-time-optimizing-drop-times-using-machine-learning-d3f6fb1b0f31)

2. **Netflix** (Media and Entertainment, Streaming Services)

- [Supporting Diverse ML Systems at Netflix](https://netflixtechblog.com/supporting-diverse-ml-systems-at-netflix-2d2e6b6d205d)
- [Optimizing the Netflix Streaming Experience with Data Science](http://techblog.netflix.com/2014/06/optimizing-netflix-streaming-experience.html)
- [Netflix Recommendations: Beyond the 5 stars (Part 1)](http://techblog.netflix.com/2012/04/netflix-recommendations-beyond-5-stars.html)
- [Machine Learning Platform - Netflix Research](https://research.netflix.com/research-area/machine-learning-platform)
- [Evolution of ML Fact Store](https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762)
- [Scaling Media Machine Learning at Netflix](https://netflixtechblog.com/scaling-media-machine-learning-at-netflix-f19b400243)
- [InTune: Reinforcement Learning-based Data Pipeline Optimization for Deep Learning Recommendation Models](https://research.netflix.com/publication/intune-reinforcement-learning-based-data-pipeline-optimization-for-deep)

3. **Uber** (Transportation and Technology)

- [Real-time Data Infrastructure at Uber](https://arxiv.org/pdf/2104.0087.pdf)
- [Michelangelo: Uber's Machine Learning Platform](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
- [Building Scalable Streaming Pipelines for Near Real-Time Features](https://www.uber.com/en-IN/blog/building-scalable-streaming-pipelines/)
- [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/blog/deepeta-how-uber-predicts-arrival-times/)
- [Palette Feature Store: Uber's Centralized Feature Management Platform](https://www.uber.com/en-IN/blog/palette-meta-store-journey/)
- [Project RADAR: Intelligent Early Fraud Detection](https://www.uber.com/en-GB/blog/project-radar-intelligent-early-fraud-detection/)
- [How Uber Optimizes the Timing of Push Notifications using ML and Linear Programming](https://www.uber.com/en-IN/blog/how-uber-optimizes-push-notifications-using-ml/)
- [Uber's Real-Time Document Check](https://www.uber.com/en-GB/blog/ubers-real-time-document-check/)
- [Personalized Marketing at Scale: Uber's Out-of-App Recommendation System](https://www.uber.com/en-GB/blog/personalized-marketing-at-scale/)
- [Stopping Uber Fraudsters Through Risk Challenges](https://www.uber.com/en-GB/blog/stopping-uber-fraudsters-through-risk-challenges/)

4. **TikTok** (Social Media, Entertainment, and Technology)

- [Monolith: Real-Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663)
- [Build TikTok's Personalized Real-Time Recommendation System in Python](https://2024.pycon.de/program/DPGRGW/)
- [Real-time Data Processing at TikTok](https://www.techaheadcorp.com/blog/decoding-tiktok-system-design-architecture/)
- [The AI Algorithm that got TikTok Users Hooked](https://www.argoid.ai/blog/the-ai-algorithm-that-got-tiktok-users-hooked)

5. **Meta** (Technology, Social Media, and Artificial Intelligence)

- [Spiral: Self-tuning services via real-time machine learning](https://engineering.fb.com/2018/06/28/data-infrastructure/spiral-self-tuning-services-via-real-time-machine-learning/)
- [Scaling data ingestion for machine learning training at Meta](https://engineering.fb.com/2022/09/19/ml-applications/data-ingestion-machine-learning-training-meta/)
- [Meta's approach to machine learning prediction robustness](https://engineering.fb.com/2024/07/10/data-infrastructure/machine-learning-ml-prediction-robustness-meta/)
- [Machine Learning Platform - Meta Research](https://ai.meta.com/research/areas/machine-learning-platform/)
- [Inside Meta's AI optimization platform for engineers across the company](https://ai.meta.com/blog/looper-meta-ai-optimization-platform-for-engineers/)
- [New AI advancements drive Meta's ads system performance and efficiency](https://ai.meta.com/blog/ai-ads-performance-efficiency-meta-lattice/)
- [AI debugging at Meta with HawkEye](https://engineering.fb.com/2023/12/19/data-infrastructure/hawkeye-ai-debugging-meta/)
- [How machine learning powers Facebook's News Feed ranking algorithm](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- [Scaling the Instagram Explore recommendations system](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)

6. **Google** (Technology, Internet Services, and Artificial Intelligence)

- [Real-time ML analysis with TensorFlow, BigQuery, and Redpanda](https://www.redpanda.com/blog/real-time-machine-learning-processing-tensorflow-bigquery)
- [Real-time AI with Google Cloud Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai)
- [Streaming analytics solutions | Google Cloud](https://cloud.google.com/solutions/stream-analytics)
- [Introduction to Vertex AI Feature Store](https://cloud.google.com/vertex-ai/docs/featurestore/overview)
- [Real-time Data Infrastructure at Google](https://arxiv.org/pdf/2104.0087.pdf)
- [How Google Search ranking works](https://searchengineland.com/how-google-search-ranking-works-445141)

7. **Spotify** (Music Streaming, Technology, and Entertainment)

- [Unleashing ML Innovation at Spotify with Ray](https://engineering.atspotify.com/2023/02/unleashing-ml-innovation-at-spotify-with-ray/)
- [Product Lessons from ML Home: Spotify's One-Stop Shop for Machine Learning](https://engineering.atspotify.com/2022/01/product-lessons-from-ml-home-spotifys-one-stop-shop-for-machine-learning/)
- [The Winding Road to Better Machine Learning Infrastructure Through Tensorflow Extended and Kubeflow](https://engineering.atspotify.com/2019/12/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/)
- [Feature Stores at Spotify: Building & Scaling a Centralized Platform](https://www.tecton.ai/apply/session-video-archive/feature-stores-at-spotify-building-scaling-a-centralized-platform/)
- [Real-time Data Infrastructure at Spotify](https://arxiv.org/pdf/2104.0087.pdf)
- [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part I)](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-i/)
- [The Rise (and Lessons Learned) of ML Models to Personalize Content on Home (Part II)](https://engineering.atspotify.com/2021/11/the-rise-and-lessons-learned-of-ml-models-to-personalize-content-on-home-part-ii/)

8. **Instacart** (E-commerce, Grocery Delivery, Technology)

- [Lessons Learned: The Journey to Real-Time Machine Learning at Instacart](https://www.instacart.com/company/how-its-made/lessons-learned-the-journey-to-real-time-machine-learning-at-instacart/)
- [How Instacart Modernized the Prediction of Real Time Availability for Hundreds of Millions of Items](https://tech.instacart.com/how-instacart-modernized-the-prediction-of-real-time-availability-for-hundreds-of-millions-of-items-59b2a82c89fe)
- [Predicting the real-time availability of 200 million grocery items](https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe)
- [Griffin: How Instacart's ML Platform Tripled ML Applications in a year](https://www.instacart.com/company/how-its-made/griffin-how-instacarts-ml-platform-tripled-ml-applications-in-a-year/)
- [How Instacart Used Data Streaming to Meet COVID-19 Challenges](https://www.confluent.io/customers/instacart/)
- [Distributed Machine Learning at Instacart](https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423)
- [Instacart's Item Availability Architecture: Solving for scale and consistency](https://tech.instacart.com/instacarts-item-availability-architecture-solving-for-scale-and-consistency-f5661acb20a6)
- [Supercharging ML/AI Foundations at Instacart](https://tech.instacart.com/supercharging-ml-ai-foundations-at-instacart-d48214a2b511)
- [Real-time Fraud Detection with Yoda and ClickHouse](https://tech.instacart.com/real-time-fraud-detection-with-yoda-and-clickhouse-bd08e9dbe3f4)

9. **DoorDash** (Food Delivery, Technology, and Logistics)

- [Building Scalable Real-Time Event Processing with Kafka and Flink](https://doordash.engineering/2022/08/02/building-scalable-real-time-event-processing-with-kafka-and-flink/)
- [How DoorDash Built an Ensemble Learning Model for Time Series Forecasting](https://careersatdoordash.com/blog/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/)
- [Building a Gigascale ML Feature Store with Redis](https://careersatdoordash.com/blog/building-a-gigascale-ml-feature-store-with-redis/)
- [Using ML and Optimization to Solve DoorDash's Dispatch Problem](https://careersatdoordash.com/blog/using-ml-and-optimization-to-solve-doordashs-dispatch-problem/)
- [Maintaining Machine Learning Model Accuracy Through Monitoring](https://careersatdoordash.com/blog/monitor-machine-learning-model-drift/)
- [Building Riviera: A Declarative Real-Time Feature Engineering Framework](https://careersatdoordash.com/blog/building-a-declarative-real-time-feature-engineering-framework/)
- [Engineering Systems for Real-Time Predictions @DoorDash](https://www.infoq.com/presentations/doordash-real-time-predictions/)
- [How DoorDash Upgraded a Heuristic with ML to Save Thousands of Canceled Orders](https://careersatdoordash.com/blog/how-doordash-upgraded-a-heuristic-with-ml-to-save-thousands-of-canceled-orders/)
- [Personalizing the DoorDash Retail Store Page Experience](https://careersatdoordash.com/blog/personalizing-the-doordash-retail-store-page-experience/)
- [Homepage Recommendation with Exploitation and Exploration](https://careersatdoordash.com/blog/homepage-recommendation-with-exploitation-and-exploration/)
- [Managing Supply and Demand Balance Through Machine Learning](https://careersatdoordash.com/blog/managing-supply-and-demand-balance-through-machine-learning/)
- [3 Changes to Expand DoorDash's Product Search Beyond Delivery](https://careersatdoordash.com/blog/3-changes-to-expand-doordashs-product-search/)
- [Improving ETAs with multi-task models, deep learning, and probabilistic forecasts](https://careersatdoordash.com/blog/improving-etas-with-multi-task-models-deep-learning-and-probabilistic-forecasts/)
- [Beyond the Click: Elevating DoorDash's personalized notification experience with GNN recommendation](https://careersatdoordash.com/blog/doordash-customize-notifications-how-gnn-work/)

10. **Booking.com** (Travel and Technology)

- [Booking.com Launches New AI Trip Planner to Enhance Travel Planning Experience](https://news.booking.com/bookingcom-launches-new-ai-trip-planner-to-enhance-travel-planning-experience/)
- [Booking.com Enhances Travel Planning with New AI-Powered Features for Easier, Smarter Decisions](https://news.booking.com/bookingcom-enhances-travel-planning-with-new-ai-powered-features--for-easier-smarter-decisions/)
- [Machine Learning in production: the Booking.com approach](https://booking.ai/https-booking-ai-machine-learning-production-3ee8fe943c70)
- [Booking.com: Building a Scalable Machine Learning Platform](https://h2o.ai/case-studies/booking-com/)
- [Booking.com accelerates data analysis with Confluent Cloud](https://www.confluent.io/customers/booking-com/)
- [Leverage graph technology for real-time Fraud Detection and Prevention](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5)

11. **Grab** (Technology, Ride-Hailing, Food Delivery, and Digital Payments)

- [Real-time data ingestion in Grab](https://engineering.grab.com/real-time-data-ingestion)
- [How we got more accurate estimated time-of-arrivals in the app while pushing down tech costs](https://www.grab.com/sg/inside-grab/how-we-got-more-accurate-estimated-time-of-arrivals-in-the-app-while-pushing-down-tech-costs/)
- [GrabRideGuide, our new AI tool that predicts ride demand hotspots](https://www.grab.com/sg/inside-grab/stories/grabrideguide-our-new-ai-tool-that-predicts-high-demand-areas/)
- [Evolution of Catwalk: Model serving platform at Grab](https://engineering.grab.com/catwalk-evolution)
- [Grab App at Scale with ScyllaDB](https://www.scylladb.com/2021/06/23/grab-app-at-scale-with-scylla/)
- [Harnessing AI for public good: Grab's approach to AI Governance](https://www.grab.com/sg/inside-grab/stories/harnessing-ai-for-public-good-grabs-approach-to-ai-governance/)
- [Unsupervised graph anomaly detection - Catching new fraudulent behaviours](https://engineering.grab.com/graph-anomaly-model)

12. **Didact AI** (Finance, Machine Learning, Stock Trading)

- [Didact AI: The anatomy of an ML-powered stock picking engine](https://principiamundi.com/posts/didact-anatomy/)

13. **Glassdoor** (Technology, Job Search and Company Reviews)

- [Glassdoor uses machine learning to tell users if they're being paid fairly](https://www.zdnet.com/article/glassdoor-uses-machine-learning-to-tell-users-if-theyre-being-paid-fairly/)

13. **Instacart** (E-commerce, Grocery Delivery, Technology)

- [Lessons Learned: The Journey to Real-Time Machine Learning at Instacart](https://www.instacart.com/company/how-its-made/lessons-learned-the-journey-to-real-time-machine-learning-at-instacart/)
- [How Instacart Modernized the Prediction of Real Time Availability for Hundreds of Millions of Items](https://tech.instacart.com/how-instacart-modernized-the-prediction-of-real-time-availability-for-hundreds-of-millions-of-items-59b2a82c89fe)
- [Predicting the real-time availability of 200 million grocery items](https://tech.instacart.com/predicting-real-time-availability-of-200-million-grocery-items-in-us-canada-stores-61f43a16eafe)
- [Griffin: How Instacart's ML Platform Tripled ML Applications in a year](https://www.instacart.com/company/how-its-made/griffin-how-instacarts-ml-platform-tripled-ml-applications-in-a-year/)
- [How Instacart Used Data Streaming to Meet COVID-19 Challenges](https://www.confluent.io/customers/instacart/)
- [Distributed Machine Learning at Instacart](https://tech.instacart.com/distributed-machine-learning-at-instacart-4b11d7569423)
- [Instacart’s Item Availability Architecture: Solving for scale and consistency](https://tech.instacart.com/instacarts-item-availability-architecture-solving-for-scale-and-consistency-f5661acb20a6)
- [Supercharging ML/AI Foundations at Instacart](https://tech.instacart.com/supercharging-ml-ai-foundations-at-instacart-d48214a2b511)
- [Real-time Fraud Detection with Yoda and ClickHouse](https://tech.instacart.com/real-time-fraud-detection-with-yoda-and-clickhouse-bd08e9dbe3f4)

14. **DoorDash** (Food Delivery, Technology, and Logistics)

- [Building Scalable Real-Time Event Processing with Kafka and Flink](https://doordash.engineering/2022/08/02/building-scalable-real-time-event-processing-with-kafka-and-flink/)
- [How DoorDash Built an Ensemble Learning Model for Time Series Forecasting](https://careersatdoordash.com/blog/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/)
- [Building a Gigascale ML Feature Store with Redis](https://careersatdoordash.com/blog/building-a-gigascale-ml-feature-store-with-redis/)
- [Using ML and Optimization to Solve DoorDash's Dispatch Problem](https://careersatdoordash.com/blog/using-ml-and-optimization-to-solve-doordashs-dispatch-problem/)
- [Maintaining Machine Learning Model Accuracy Through Monitoring](https://careersatdoordash.com/blog/monitor-machine-learning-model-drift/)
- [Building Riviera: A Declarative Real-Time Feature Engineering Framework](https://careersatdoordash.com/blog/building-a-declarative-real-time-feature-engineering-framework/)
- [Engineering Systems for Real-Time Predictions @DoorDash](https://www.infoq.com/presentations/doordash-real-time-predictions/)
- [How DoorDash Upgraded a Heuristic with ML to Save Thousands of Canceled Orders](https://careersatdoordash.com/blog/how-doordash-upgraded-a-heuristic-with-ml-to-save-thousands-of-canceled-orders/)
- [Personalizing the DoorDash Retail Store Page Experience](https://careersatdoordash.com/blog/personalizing-the-doordash-retail-store-page-experience/)
- [Homepage Recommendation with Exploitation and Exploration](https://careersatdoordash.com/blog/homepage-recommendation-with-exploitation-and-exploration/)
- [Managing Supply and Demand Balance Through Machine Learning](https://careersatdoordash.com/blog/managing-supply-and-demand-balance-through-machine-learning/)
- [3 Changes to Expand DoorDash’s Product Search Beyond Delivery](https://careersatdoordash.com/blog/3-changes-to-expand-doordashs-product-search/)
- [Improving ETAs with multi-task models, deep learning, and probabilistic forecasts](https://careersatdoordash.com/blog/improving-etas-with-multi-task-models-deep-learning-and-probabilistic-forecasts/)
- [Beyond the Click: Elevating DoorDash’s personalized notification experience with GNN recommendation](https://careersatdoordash.com/blog/doordash-customize-notifications-how-gnn-work/)

15. **Booking.com** (Travel and Technology)

- [Booking.com Launches New AI Trip Planner to Enhance Travel Planning Experience](https://news.booking.com/bookingcom-launches-new-ai-trip-planner-to-enhance-travel-planning-experience/)
- [Booking.com Enhances Travel Planning with New AI-Powered Features for Easier, Smarter Decisions](https://news.booking.com/bookingcom-enhances-travel-planning-with-new-ai-powered-features--for-easier-smarter-decisions/)
- [Machine Learning in production: the Booking.com approach](https://booking.ai/https-booking-ai-machine-learning-production-3ee8fe943c70)
- [Booking.com: Building a Scalable Machine Learning Platform](https://h2o.ai/case-studies/booking-com/)
- [Booking.com accelerates data analysis with Confluent Cloud](https://www.confluent.io/customers/booking-com/)
- [Leverage graph technology for real-time Fraud Detection and Prevention](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5)

16. **Grab** (Technology, Ride-Hailing, Food Delivery, and Digital Payments)

- [Real-time data ingestion in Grab](https://engineering.grab.com/real-time-data-ingestion)
- [How we got more accurate estimated time-of-arrivals in the app while pushing down tech costs](https://www.grab.com/sg/inside-grab/how-we-got-more-accurate-estimated-time-of-arrivals-in-the-app-while-pushing-down-tech-costs/)
- [GrabRideGuide, our new AI tool that predicts ride demand hotspots](https://www.grab.com/sg/inside-grab/stories/grabrideguide-our-new-ai-tool-that-predicts-high-demand-areas/)
- [Evolution of Catwalk: Model serving platform at Grab](https://engineering.grab.com/catwalk-evolution)
- [Grab App at Scale with ScyllaDB](https://www.scylladb.com/2021/06/23/grab-app-at-scale-with-scylla/)
- [Harnessing AI for public good: Grab's approach to AI Governance](https://www.grab.com/sg/inside-grab/stories/harnessing-ai-for-public-good-grabs-approach-to-ai-governance/)
- [Unsupervised graph anomaly detection - Catching new fraudulent behaviours](https://engineering.grab.com/graph-anomaly-model)

17. **Didact AI** (Finance, Machine Learning, Stock Trading)

- [Didact AI: The anatomy of an ML-powered stock picking engine](https://principiamundi.com/posts/didact-anatomy/) 

18. **Glassdoor** (Technology, Job Search and Company Reviews)

 - [Glassdoor uses machine learning to tell users if they're being paid fairly](https://www.zdnet.com/article/glassdoor-uses-machine-learning-to-tell-users-if-theyre-being-paid-fairly/)

19. **Booking.com** (Travel and Technology)

- [Booking.com Launches New AI Trip Planner to Enhance Travel Planning Experience](https://news.booking.com/bookingcom-launches-new-ai-trip-planner-to-enhance-travel-planning-experience/)
- [Booking.com Enhances Travel Planning with New AI-Powered Features for Easier, Smarter Decisions](https://news.booking.com/bookingcom-enhances-travel-planning-with-new-ai-powered-features--for-easier-smarter-decisions/)
- [Machine Learning in production: the Booking.com approach](https://booking.ai/https-booking-ai-machine-learning-production-3ee8fe943c70)
- [Booking.com: Building a Scalable Machine Learning Platform](https://h2o.ai/case-studies/booking-com/)
- [Booking.com accelerates data analysis with Confluent Cloud](https://www.confluent.io/customers/booking-com/)
- [Leverage graph technology for real-time Fraud Detection and Prevention](https://medium.com/booking-com-development/leverage-graph-technology-for-real-time-fraud-detection-and-prevention-438336076ea5)

20. **Grab** (Technology, Ride-Hailing, Food Delivery, and Digital Payments)

- [Real-time data ingestion in Grab](https://engineering.grab.com/real-time-data-ingestion)
- [How we got more accurate estimated time-of-arrivals in the app while pushing down tech costs](https://www.grab.com/sg/inside-grab/how-we-got-more-accurate-estimated-time-of-arrivals-in-the-app-while-pushing-down-tech-costs/)
- [GrabRideGuide, our new AI tool that predicts ride demand hotspots](https://www.grab.com/sg/inside-grab/stories/grabrideguide-our-new-ai-tool-that-predicts-high-demand-areas/)
- [Evolution of Catwalk: Model serving platform at Grab](https://engineering.grab.com/catwalk-evolution)
- [Grab App at Scale with ScyllaDB](https://www.scylladb.com/2021/06/23/grab-app-at-scale-with-scylla/)
- [Harnessing AI for public good: Grab's approach to AI Governance](https://www.grab.com/sg/inside-grab/stories/harnessing-ai-for-public-good-grabs-approach-to-ai-governance/)
- [Unsupervised graph anomaly detection - Catching new fraudulent behaviours](https://engineering.grab.com/graph-anomaly-model)

21. **Didact AI** (Finance, Machine Learning, Stock Trading)

- [Didact AI: The anatomy of an ML-powered stock picking engine](https://principiamundi.com/posts/didact-anatomy/) 

22. **Glassdoor** (Technology, Job Search and Company Reviews)

- [Glassdoor uses machine learning to tell users if they're being paid fairly](https://www.zdnet.com/article/glassdoor-uses-machine-learning-to-tell-users-if-theyre-being-paid-fairly/)

23. **Dailymotion** (Video Sharing and Streaming, AdTech)

- [Case study Dailymotion - Martech Compass](https://martechcompass.com/case-study-dailymotion/)
- [AI Video Intelligence: Innovation for Publishers & Broadcasters](https://pro.dailymotion.com/en/blog/ai-video-intelligence/)
- [Dailymotion's Journey to Crafting the Ultimate Content-Driven Video Recommendation Engine with Qdrant Vector Database](https://qdrant.tech/blog/case-study-dailymotion/)
- [See Real time activity - Dailymotion Help Center](https://faq.dailymotion.com/hc/en-us/articles/360016890140-See-Real-time-activity)
- [How Dailymotion reinvented itself to create a better user experience](https://www.builtinnyc.com/articles/spotlight-working-at-dailymotion-tech-innovation)

24. **Coupang** (E-commerce, Technology, and Logistics)

- [A symphony of people and technology: An inside look at a Coupang fulfillment center](https://www.aboutcoupang.com/English/news/news-details/2022/A-symphony-of-people-and-technology-An-inside-look-inside-a-Coupang-fulfillment-center/default.aspx)
- [How Coupang Conquered South Korean E-commerce](https://quartr.com/insights/company-research/how-coupang-conquered-south-korean-ecommerce)
- [Matching duplicate items to improve catalog quality](https://medium.com/coupang-engineering/matching-duplicate-items-to-improve-catalog-quality-ca4abc827f94)
- [Overcoming food delivery challenges with data science](https://medium.com/coupang-engineering/overcoming-food-delivery-challenges-with-data-science-6420cac1d59)

25. **Slack** (Technology, Communication, and Collaboration Software)

- [How Slack sends Millions of Messages in Real Time](https://blog.quastor.org/p/slack-sends-millions-messages-real-time)
- [Slack delivers native and secure generative AI powered by Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/slack-delivers-native-and-secure-generative-ai-powered-by-amazon-sagemaker-jumpstart/)
- [Real-Time Messaging Architecture at Slack](https://www.infoq.com/news/2023/04/real-time-messaging-slack/)
- [How We Built Slack AI To Be Secure and Private](https://slack.engineering/how-we-built-slack-ai-to-be-secure-and-private/)
- [Real Time Messaging API | Node Slack SDK](https://tools.slack.dev/node-slack-sdk/rtm-api/)
- [Privacy principles: search, learning and artificial intelligence](https://slack.com/intl/en-in/trust/data-management/privacy-principles)
- [Email Classification](https://slack.engineering/email-classification/)

26. **Swiggy** (Food Delivery, Technology, and E-commerce)

- [Building Rock-Solid ML Systems](https://bytes.swiggy.com/building-rock-solid-ml-systems-d1d5ab544f0e)
- [Swiggy's Generative AI Journey: A Peek Into the Future](https://bytes.swiggy.com/swiggys-generative-ai-journey-a-peek-into-the-future-2193c7166d9a)
- [Swiggylytics: Swiggy's real-time Analytics SDK](https://bytes.swiggy.com/swiggylytics-5046978965dc)
- [Hyperlocal Forecasting at Scale: The Swiggy Forecasting platform](https://bytes.swiggy.com/hyperlocal-forecasting-at-scale-the-swiggy-forecasting-platform-c07ecd5f5b86)
- [Enabling data science at scale at Swiggy: the DSP story](https://bytes.swiggy.com/enabling-data-science-at-scale-at-swiggy-the-dsp-story-208c2d85faf9)
- [Deploying deep learning models at scale at Swiggy: Tensorflow Serving on DSP](https://bytes.swiggy.com/deploying-deep-learning-models-at-scale-at-swiggy-tensorflow-serving-on-dsp-ad5da40f7a6c)
- [An ML approach for routing payment transactions](https://bytes.swiggy.com/an-ml-approach-for-routing-payment-transactions-5a14efb643a8)
- [How ML Powers — When is my order coming?](https://bytes.swiggy.com/how-ml-powers-when-is-my-order-coming-part-ii-eae83575e3a9)

27. **Nubank** (Financial Technology (Fintech), Digital Banking)

- [Presenting Precog, Nubank's Real Time Event AI](https://building.nubank.com.br/presenting-precog-nubanks-real-time-event-ai/)
- [Real-time machine learning models in real life](https://building.nubank.com.br/real-time-machine-learning-models-in-real-life/)
- [Fklearn: Nubank's machine learning library](https://building.nubank.com.br/introducing-fklearn-nubanks-machine-learning-library-part-i-2/)
- [The potential of sequential modeling in fraud prevention](https://building.nubank.com.br/the-potential-of-sequential-modeling-in-fraud-prevention-insights-from-experts-at-nubank/)
- [Nubank acquires Hyperplane to accelerate AI-first strategy](https://www.fintechfutures.com/2024/06/brazils-nubank-buys-ai-powered-data-intelligence-start-up-hyperplane/)
- [Beyond prediction machines](https://building.nubank.com.br/beyond-prediction-machines/)


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
