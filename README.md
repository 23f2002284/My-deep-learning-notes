>"Slow and steady wins the race".
Don't let the volume scare you. We build this brick by brick. Trust the process, and trust yourself.*

# Projects ( to learn )
0. Build a Neural Network from Scratch (using only NumPy)
1. MNIST
	1. MNIST : Pytorch DataLoader [Link](https://www.deep-ml.com/labs/1)
	2. MNIST Design-Your-Own tiny Pytorch Model: [Link](https://www.deep-ml.com/labs/2)
	3. MNIST: Design Your Own Pytorch Optimizer [Link](https://www.deep-ml.com/labs/3)
	4. MNIST Classification Loss (with Gradient): [Link](https://www.deep-ml.com/labs/4)
	5. MNIST : Adversarial Example Generation : [Link](https://www.deep-ml.com/labs/5)
	6. MNIST: Build Neural Network from Scratch (NumPy Only) : [Link](https://www.deep-ml.com/labs/6)
	7. MNIST Fix Very Deep Network Training: [Link](https://www.deep-ml.com/labs/7)
2. Design Your Own Optimizer ( Numpy ) : [Link](https://www.deep-ml.com/labs/8)
3. Design Your Own Activation Function : [Link](https://www.deep-ml.com/labs/9)
4. CIFAR-10 Image Recognizer [Link](https://github.com/RubixML/CIFAR-10)
5. Color Clusterer : Kmeans [Link](https://github.com/RubixML/Colors)
6. Credit Card Default Predictor : Logistic regression, t-SNE [Link](https://github.com/RubixML/Credit)
7. Customer Churn Predictor : Naive Bayes [Link](https://github.com/RubixML/Churn)
8. DNA Taxonomer : Large dataset  and not well defined data [Link](https://github.com/RubixML/DNA)
9. Dota 2 Game Outcome Predictor : Naive Bayes and heavily feature engineering than ML [Link](https://github.com/RubixML/Dota2)
10. (Sorry) Divorce Predictor : KNN [Link](https://github.com/RubixML/Divorce)
11. Human Activity Recognizer : Sensor Data , Softmax Classifier [Link](https://github.com/RubixML/HAR)
12. House Price Prediction : Gradient Boosted Machine [Link](https://github.com/RubixML/Housing)
13. Iris Flower Classifier : KNN [Link](https://github.com/RubixML/Iris)
14. Text Sentiment Analyzer : feed-forward neural network [Link](https://github.com/RubixML/Sentiment)
15. Titanic Survival Prediction : to introduce Kaggle [Link](https://github.com/Jenutka/titanic_php)
16. Time Series Forecasting
	1. Stock price prediction ( LSTM / GRU)
	2. Weather forecasting ( transformer -based )
17. Object detection
	1. YOLO (You Only Look Once) on COCO dataset
	2. Faster R-CNN implementation
18. Image Segmentation
	1. U-Net for medical image segmentation
	2. Semantic segmentation on Cityscapes
19. Paper Implementation
	1. Transformer from Scratch
	2. encoding
	3. different architecture
20. Multi-modal projects
	1. CLIP-style Image-text matching
	2. Vision Language model (e.g. BLIP)
21. Deploy a Model as API:
   - FastAPI + Docker
   - Model serving with TorchServe or BentoML
22. Fashion-MNIST, EMIST Dataset projects
23. Generating Handwritten Digits : Generative Adversarial network 
24. Association analysis : apriori
25. Image Reconstruction With RBM : restricted boltmann machine 
26. Genetic Algorithm
		**Description:** Implementation of a Genetic Algorithm which aims to product the user specified target string. This implementation calculates each candidate's fitness based on the alphabetical distance between the candidate and the target. A candidate is selected as a parent with probabilities proportional to the candidate's fitness. Reproduction is implemented as a single-point crossover between pairs of parents. Mutation is done by randomly assigning new characters with uniform probability.
27. Classification With CNN
28. Density-Based Clustering
29. Evolutionary Evolved Neural Network
30. Given an image (and its label value) predict the sum of the digits in the image. [Link](https://github.com/MLNS-Deep-Learning-Assignment/mlns-deep-learning-assignment-classroom-spring2025-deep-learning-assignment-deep-learning-assignment)
31. Problem Solving : [Link](https://github.com/xbresson/CS5242_2025/blob/main/labs_lecture02/lab02_pytorch_tensor1/pytorch_tensor_part1_exercise.ipynb)
32. Style transfer
33. Adversarial Attacks: Fooling your own network with noise.
34. all project in each segments as mentioned

# Case Studies ( to learn )
## Kaggle Classic
1. **Rossmann Store Sales** ( Time Series and structured data )
	"**Entity Embeddings:** This was one of the first major proofs that Neural Networks could beat XGBoost on tabular data if you used "Entity Embeddings" for categorical variables. The 3rd place solution paper is a classic."
2. **Porto Seguro’s Safe Driver Prediction** ( Handling Imbalanced Data & Noise )
	"**Denoising Autoencoders (DAE):** The winning solution by Michael Jahrer showed how to use DAEs to clean noisy insurance data before feeding it into a classifier. It teaches you that "better data > better models."
3. **Home Credit Default Risk** (Feature Engineering)
	"**Domain Knowledge Construction:** The winners didn't just use the raw data; they built features based on financial logic (e.g., debt-to-income ratios) and aggregated massive relational tables. Read the "LightGBM + LSTM" hybrid approaches here."
4. **Google Landmark Retrieva**(Computer Vision)
	"**Metric Learning:** Unlike simple classification (Is this a cat?), this teaches you "retrieval"—finding similar images in a massive database. You learn about ArcFace, CosFace, and ranking losses."
5. **Jigsaw Toxic Comment Classification**(NLP & Bias)
	"**BERT & Transformers:** This is the playground where the industry moved from LSTM/GRU to Transformers. It highlights the difficulty of defining "toxicity" and dealing with multi-label classification."
### few niche and popular Case study 
1. The Netflix Recommender System (The "Granddaddy" of Case Studies) [Link](https://netflixtechblog.com/)
2. Uber’s "Michelangelo" (MLOps & Feature Stores) [Link](https://www.uber.com/blog/michelangelo-machine-learning-platform/)
3. Airbnb Search Ranking (Balancing Markets) [Link](https://medium.com/airbnb-engineering)
4. Pinterest "PinSage" (Graph Neural Networks) [Link](https://medium.com/pinterest-engineering)
5. Instacart Market Basket Analysis
6. Spotify’s "Algotorial" Playlists
7. Otto Group Product Classification
	"a textbook example of a structured-data competition where clever stacking and careful CV beat big complex models."
8. ImageNet / AlexNet -> ResNet history (competitive CV case study).
	"Why: shows how a single good idea (deep conv nets + ReLU + GPUs for AlexNet, then residual connections for ResNet) transformed the entire field. Learn how research breakthroughs map to practical performance gains in competitions.  
	Read for: architecture design principles, transfer learning, large-scale dataset lessons."
### Niche Hackathons and their winning blogs as casestudy
1. 26th place in "# Predicting the Beats-per-Minute of Songs" : [Link](https://www.kaggle.com/competitions/playground-series-s5e9/writeups/26th-place-fe-pseudo-labels-residuals)
2. Public 3rd Place Solution "# NFL Big Data Bowl 2026 - Prediction (Predict player movement while the ball is in the air) " : [Link](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
3. ( not a solution blogs ) Kaggle messed up this dataset on purpose "# Diabetes Prediction Challenge": [Link](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/652262) another one: [Link](https://www.kaggle.com/competitions/playground-series-s5e12/discussion/651066)
4. NVARC solution "# ARC Prize 2025: Create an AI capable of novel reasoning" : [Link](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc)
5. All solutions of "# Google - The Gemma 3n Impact Challenge": [Link](https://www.kaggle.com/competitions/google-gemma-3n-hackathon/discussion/657756)
6. ( not a solution blogs ) How to write a Solution write ups: [Link](https://www.kaggle.com/solution-write-up-documentation)
### points to note
- **Don't Re-implement Everything:** You don't have Uber's data. Instead, read their "Architecture" diagrams. Draw them out on paper. Ask: * "Why did they separate the offline training from online serving?"*

- **Focus on the "Why", not the Code:** Code rots; logic stays. The "Entity Embeddings" concept from Rossmann (2015) is still used in LLMs today.

- **The "Data Leakage" Hunt:** In every Kaggle discussion, look for threads about "Leakage." Learning how seasoned pros spot that _the answer was accidentally hidden in the ID column_ will save your career one day.

# Case Studies ( to solve )
have to add on the go ;)

# Foundations and Core ML
### Maths
Learn while covering syllabus parallely
1. **Linear Algebra:** Eigenvalues, SVD, matrix calculus
2. **Calculus:** Gradients, partial derivatives, chain rule
3. **Probability & Statistics:** Bayes theorem, distributions, hypothesis testing
4. **Information Theory:** Entropy, KL divergence, cross-entropy
5. **Optimization Theory:** Gradient descent variants, convex optimization
### Supervised Learning
1. Adaboost
2. Bayesian Regression
3. Decision Tree
4. Elastic Net
5. Gradient Boosting
6. K Nearest Neighbors
7. Lasso Regression
8. Linear Discriminant Analysis
9. Linear regression
10. Logistic regression
11. Multi-class Linear Discriminant analysis
12. Multilayer perceptron
13. Naive Bayes
14. Neuroevolution
15. Particle swarm Optimization of neural network
16. Perceptron
17. Polynomial regression
18. Random Forest
19. Ridge regression
20. Support Vector machine
21. XGBoost
22. 12. CatBoost
23. LightGBM
24. Isolation Forest ( anomaly detection )
25. t-SNE and UMAP ( dimensionality reduction, visualization)
26. HDBSCAN ( density based clustering, better than DBSCAN )
### Unsupervised Learning
1. Apriori
2. Autoencoder
3. DBSCAN
4. FP-Growth
5. Gaussian Mixture Model
6. Generative Adversarial Network
7. Genetic algorithm
8. K-Means
9. Partitioning Around Medoids
10. Principle Component Analysis
11. Restricted Boltzmann machine


# Deep Learning Fundamentals
1. Neural Network [Link](https://github.com/23f2002284/My-deep-learning-notes/blob/main/Deep%20Learning/Neural-Networks-%20Index.pdf)
2. Backpropagation
3. MLPs
4. Layers
	1. Attention Layer
	2. Layer Normalization (vs Batch Normalization )
	3. Group Normalization
	4. Residual Connections / Skip Connections
	5. Depthwise separable Convolutions ( MobileNet )
5. Loss functions
	1. Cross entropy ( Binary, Categorical, Sparse )
	2. Focal Loss ( for imbalanced Data )
	3. Contrastive Loss ( Siamese networks )
	4. Triplet Loss (face recognition)
	5. Huber Loss (robust regression)
6. Optimizers
	1. SGD with momentum
	2. Adam, AdamW ( weight decay fix )
	3. RMSprop
	4. AdaGrad, AdaDelta
	5. Lion
	6. Learning rate schedulers
		1. Cosine Annealing
		2. OneCycleLR
		3. ReduceLROnPlateau
7. Gradient Clipping, Mixed precision training, Distributed training
8. Regularizations
	1. Dropout
	2. ...

# Computer Vision
1. Image Classification
	1. AlexNet
	2. VGG
	3. ResNet
	4. EfficientNet
	5. Vision Transformer ( ViT )
2. Object Detection
	1. R-CNN family
	2. YOLO
	3. DETR
3. Segmentation
	1. semantic
		1. U-Net
		2. DeepLab
	2. Instance
	3. Panoptic
	4. SAM
4. Generative Models
	1. GANs (DCGAN, StyleGAN, CycleGAN, ....)
	2. Diffusion models ( DDPM, Stable Diffusion, DALL-E, ... )
5. Video Understanding
	1. Action recognition ( 3D CNNs, I3D )
	2. Video Segmentation , tracking
6. Self-supervised learning
	1. SIMCLR
	2. MoCo
	3. DINO
	4. MAE ( masked autoencoder )
7. 3D Vision
	1. NeRF ( Neural radiance Fields )
	2. Point Cloud processing ( PointNet )
8. Projects
	1. Build a face recognition system ( MTCNN + FaceNet )
	2. Real time object detection with YOLO-v11
	3. image-to-image translation ( Pix2Pix, CycleGAN )
	4. Stable Diffusion finetuning ( DreamBooth, LoRA )

# NLP Essentials
1. Text preprocessing
	1. Tokenization
		1. WordPiece
		2. BPE
		3. SentencePiece
	2. different preprocessing
		1. stopword removal, stemming, lemmatization
2. Word embeddings
	1. Word2Vec
	2. GloVe
	3. FastText
3. Sequence Models
	1. RNNs
	2. LSTMs
	3. GRUs
	4. Seq2Seq
	5. attention mechanism
4. Modern LLMs
	1. LLM portion + 
	2. Different LLMs and their case studies
5. Tasks
	1. Sentiment analysis
	2. NER
	3. text classification
	4. Question answering
	5. summarization
	6. machine translation
	7. dialogue systems
	8. OCR ( Optical character recognition )
	9. emotion recognition
6. Prompt engineering
	1. Zero-shot
	2. few-shot
	3. chain-of-thought
7. Evaluation metrics
	1. BLEU
	2. ROUGE
	3. METEOR
	4. BERTScore
	5. perplexity
8. Projects
	- Build a chatbot with GPT-2/LLaMA
	- Fine-tune BERT for classification
	- Named Entity Recognition pipeline
	- Summarization with T5/BART

# Embedding and vector databases
1. Embedding models
	1. Sentence transformers
	2. OpenAI embeddings
	3. Gemini Embeddings
	4. Cohere embeddings
	5. BGE, E5, instructor models
2. VectorDBs
	1. pinecone
	2. Weaviate
	3. Qdrant
	4. Milvus
	5. Chroma
	6. FAISS
3. Different Techniques
	1. Semantic Search
	2. Hybrid Search ( vector + keyword )
	3. Reranking ( Cohere Rerank, cross-encoders )

# Modern LLMs
1. Transformers
	1. Architecture
		1. Self Attention
		2. Multi-head Attention
		3. Encodings ( absolute, realative, RoPE, ALiBi )
		4. Feed-forward network
	2. Variants
		1. Encoder only ( BERT )
		2. Decoder only ( GPT )
		3. Encoder - Decoder ( T5 )
	3. Scaling
		1. Chinchilla scaling laws
		2. Mixture of Experts
	4. Efficient Transformers
		1. Sparse Transformers
		2. Flash Attention
		3. Linformer
		4. performers
2. Pretraining
	1. Data collection and cleaning ( like CommonCrawl, C4 )
	2. Tokenization strategies
	3. Training infrastructure ( multi-GPU, FSDP )
	4. Compute optimization ( gradient checkpointing, mixed precision )
3. Midtraining
	1. Domain Adaptation (code, medical, legal)
	2. continual learning
4. posttraining
	1. Supervised fine-tuning (SFT)
	2. RLHF (reward modeling, PPO)
	3. Direct preference Optimization(DPO)
	4. Constitutional AI
5. Fine Tuning
	1. Full finetuning
	2. Prefix Tuning
	3. Prompt Tuning
	4. IA3 ( Few Shot Parameter efficient fine tuning )
	5. Quantized Fine-tuning
	6. Tools for fine tuning
		1. Unsloth
		2. TRL
		3. LLaMA-Factory
6. With LLM API
	1. More about APIs
	2. Different Models and Different API providers
	3. Prompt engineering techniques
		1. Chain-of-Thoughts ( CoT )
		2. Tree-of-Thoughts
		3. ReAct ( Reasoning + Acting )
		4. Self-consistency
	4. Funtion Calling
	5. Structured Output
	6. Streaming Responses
	7. Cost Optimization
		1. Catching
		2. Batch Processing

# RAG
1. Document processing
	1. Chunking strategies
	2. metadata extraction
2. Query translation
	1. Query decomposition
	2. Pseudo-documents
3. Routing
	1. Logical routing
	2. semantic routing
4. Query Construction
	1. Relational DBs
	2. Graph DBs
	3. Vector DBs
5. Indexing
	1. chunk optimization
	2. multi-representation indexing
	3. specialized embeddings
	4. hierarchical indexing
6. Retrieval
	1. Ranking
	2. Refinement
	3. Active retrieval
	4. Dense retrieval
	5. Sparse retrieval
	6. Hybrid Search
7. Generation
	1. Active retrieval
8. Advanced RAG
	1. Query transformation
	2. Self-RAG
	3. Corrective RAG
	4. Agentic RAG
9. Evaluation
	1. context relevence, answer faithfulness
	2. RAGAS Framework
	3. ...
10. Projects
	1. Build a document Q&A system
	2. Code Documentation assistant
	3. Research paper assistant

# RL
1. **Fundamentals:**
   - MDP, value functions, Bellman equation
2. **Value-Based:**
   - Q-Learning, DQN, Double DQN, Dueling DQN
3. **Policy-Based:**
   - REINFORCE, A2C, A3C
4. **Actor-Critic:**
   - PPO (Proximal Policy Optimization)
   - DDPG (Deep Deterministic Policy Gradient)
   - SAC (Soft Actor-Critic)
5. **Model-Based RL:**
   - World models, Dyna-Q
6. **Multi-Agent RL**
7. **Offline RL**
8. **RLHF for LLMs** (how ChatGPT is trained!)

**Projects:**
- Train an agent for Atari games
- Robotics simulation (PyBullet)
- AlphaZero-style game AI
# Agents
1. Frameworks
	1. Langgraph
	2. crewAI
	3. LlamaIndex
	4. ...
2. Components
	1. ReAct Prompting
	2. Tool Use
	3. Planning
	4. Memory
	5. Self-reflection
3. 18 types of agent
	1. ...
	2. Multi-agent:
		1. Agent Communication protocols
		2. Debate, collaboration patterns
4. Projects
	1. Research Agent
	2. Coding Agent
	3. Customer Service Agent

# MCP
1. What is MCP ?
2. Components
	1. Servers, Client, transports
3. Integration
	1. Database Access
	2. API integration
	3. File System Access
	4. Browser Automation
4. Security
	1. OAuth & authorization
	2. Sandboxing, rate limiting

# Capstone project ( to build )
to decide on the go with this curriculum

# Upcoming Batch-1 Curriculum
1. Production
2. System Design for ML
3. Interview Prep
4. Portfolio