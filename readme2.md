### Phase 1: The Bedrock - Mathematical and Technical Foundations

#### Module 1: Linear Algebra for ML

* **🧠 Key Concepts:**
  - Scalars, vectors, matrices, and tensors as fundamental data structures
  - Matrix operations including addition, multiplication, transpose, and inverse
  - Dot product, cross product, vector norms (L1, L2)
  - Eigenvalues, eigenvectors, and matrix diagonalization
  - Orthogonality, vector projections, and singular value decomposition (SVD)

* **🎯 Learning Objectives:** Perform vector and matrix manipulations to support ML algorithms like neural networks and dimensionality reduction.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Khan Academy - Linear Algebra series](https://www.khanacademy.org/math/linear-algebra)
  * **Visual Intuition:** [3Blue1Brown - Essence of Linear Algebra playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  * **Code Implementation:** [NumPy Linear Algebra functions tutorial on Real Python](https://realpython.com/python-linear-algebra/)

* **💻 Mini-Project Idea:** Implement principal component analysis (PCA) from scratch using NumPy on the Iris dataset to reduce dimensions and visualize the results in 2D.

#### Module 2: Calculus for ML

* **🧠 Key Concepts:**
  - Limits, continuity, and differentiation rules
  - Derivatives, partial derivatives, and gradients
  - Chain rule, product rule, and quotient rule
  - Jacobians and Hessians for multivariable functions
  - Optimization techniques including gradient descent and its variants (stochastic, mini-batch)

* **🎯 Learning Objectives:** Apply calculus principles to derive and implement optimization algorithms used in training ML models.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [freeCodeCamp - Calculus for Machine Learning course](https://www.freecodecamp.org/news/deep-learning-course-math-and-applications/)
  * **Visual Intuition:** [3Blue1Brown - Essence of Calculus playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
  * **Code Implementation:** [Gradient Descent from Scratch in Python tutorial on Towards Data Science](https://towardsdatascience.com/implementing-gradient-descent-in-python-from-scratch-760a8556c31f/)

* **💻 Mini-Project Idea:** Code a gradient descent optimizer to minimize a multivariate quadratic function and plot the convergence path using Matplotlib.

#### Module 3: Probability & Statistics

* **🧠 Key Concepts:**
  - Probability axioms, conditional probability, and Bayes' theorem
  - Random variables, discrete/continuous distributions (e.g., Normal, Bernoulli, Poisson)
  - Descriptive statistics: mean, variance, standard deviation, covariance
  - Inferential statistics: hypothesis testing, p-values, and type I/II errors
  - Confidence intervals, sampling distributions, and central limit theorem

* **🎯 Learning Objectives:** Analyze datasets statistically to make data-driven decisions and validate ML model assumptions.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Khan Academy - Statistics and Probability section](https://www.khanacademy.org/math/statistics-probability)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Statistics Fundamentals playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
  * **Code Implementation:** [Introduction to Statistics in Python tutorial on DataCamp (free module)](https://www.datacamp.com/courses/introduction-to-statistics-in-python)

* **💻 Mini-Project Idea:** Conduct hypothesis testing on the Titanic dataset to compare survival rates across passenger classes using SciPy and visualize distributions.

#### Module 4: Essential Python Libraries for Data Science

* **🧠 Key Concepts:**
  - NumPy arrays, broadcasting, and vectorized operations
  - Pandas DataFrames for data manipulation, indexing, grouping, and merging
  - Matplotlib and Seaborn for data visualization (plots, histograms, heatmaps)
  - Scikit-learn basics: pipelines, estimators, and basic model fitting

* **🎯 Learning Objectives:** Manipulate, visualize, and preprocess datasets efficiently to prepare for ML workflows.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [freeCodeCamp - Python for Data Science and Machine Learning Bootcamp](https://www.freecodecamp.org/learn/machine-learning-with-python)
  * **Visual Intuition:** [Corey Schafer - Matplotlib Tutorial series](https://www.youtube.com/playlist?list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_)
  * **Code Implementation:** [Python Data Science Handbook by Jake VanderPlas (free online version)](https://jakevdp.github.io/PythonDataScienceHandbook/)

* **💻 Mini-Project Idea:** Perform exploratory data analysis (EDA) on the Boston Housing dataset, including visualizations and basic statistics, using Pandas and Seaborn.

### Phase 2: Core Machine Learning

#### Module 5: Linear & Logistic Regression

* **🧠 Key Concepts:**
  - Linear regression: ordinary least squares, cost function (MSE), and feature scaling
  - Gradient descent for parameter optimization
  - Evaluation metrics: R-squared, adjusted R-squared, MAE
  - Logistic regression: sigmoid function, log-loss cost, decision boundaries
  - Assumptions: linearity, independence, homoscedasticity, no multicollinearity

* **🎯 Learning Objectives:** Build, train, and evaluate regression models for prediction tasks while interpreting coefficients and assumptions.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Machine Learning by Andrew Ng (free audit, Weeks 1-3)](https://www.coursera.org/specializations/machine-learning-introduction)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Linear Regression and Logistic Regression videos](https://www.youtube.com/watch?v=yIYKR4sgzI8)
  * **Code Implementation:** [Scikit-learn Linear Models documentation examples](https://scikit-learn.org/stable/modules/linear_model.html)

* **💻 Mini-Project Idea:** Predict house prices using linear regression on the Boston Housing dataset and evaluate with cross-validation.

#### Module 6: K-Nearest Neighbors (KNN)

* **🧠 Key Concepts:**
  - Distance metrics: Euclidean, Manhattan, Minkowski
  - Algorithm mechanics: voting for classification, averaging for regression
  - Choosing optimal K via cross-validation
  - Curse of dimensionality and its impact on performance
  - Bias-variance tradeoff in instance-based learning

* **🎯 Learning Objectives:** Implement and tune KNN for classification and regression, understanding its lazy learning nature.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [DataCamp - Supervised Learning with scikit-learn (free KNN module)](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn)
  * **Visual Intuition:** [StatQuest with Josh Starmer - K-Nearest Neighbors video](https://www.youtube.com/watch?v=HVXime0nQeI&pp=0gcJCfwAo7VqN5tD)
  * **Code Implementation:** [Scikit-learn KNN Classifier and Regressor examples](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)

* **💻 Mini-Project Idea:** Classify iris flower species using KNN on the Iris dataset and experiment with different K values.

#### Module 7: Support Vector Machines (SVMs)

* **🧠 Key Concepts:**
  - Maximal margin hyperplanes and support vectors
  - Kernel trick for non-linear boundaries (linear, polynomial, RBF)
  - Soft margins and regularization parameter C
  - Dual formulation and Lagrange multipliers
  - Handling multi-class classification (one-vs-one, one-vs-all)

* **🎯 Learning Objectives:** Train SVM models for linear and non-linear classification, tuning hyperparameters for optimal margins.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [edX - Machine Learning with Python: A Practical Introduction (free audit, SVM section)](https://www.edx.org/learn/machine-learning/ibm-machine-learning-with-python-a-practical-introduction)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Support Vector Machines series](https://www.youtube.com/watch?v=efR1C6CvhmE)
  * **Code Implementation:** [Scikit-learn SVM documentation with examples](https://scikit-learn.org/stable/modules/svm.html)

* **💻 Mini-Project Idea:** Classify breast cancer samples as malignant or benign using SVM on the Wisconsin Breast Cancer dataset.

#### Module 8: Decision Trees & Random Forests

* **🧠 Key Concepts:**
  - Splitting criteria: entropy, information gain, Gini impurity
  - Tree pruning to prevent overfitting
  - Ensemble methods: bagging in random forests
  - Feature importance and out-of-bag error
  - Handling categorical vs. numerical features

* **🎯 Learning Objectives:** Construct tree-based models and ensembles to handle complex decision boundaries and feature interactions.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [freeCodeCamp - Machine Learning for Everybody (decision trees section)](https://www.freecodecamp.org/news/a-no-code-intro-to-the-9-most-important-machine-learning-algorithms-today/)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Decision Trees and Random Forests playlist](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&pp=0gcJCfwAo7VqN5tD)
  * **Code Implementation:** [Scikit-learn Tree-based Models documentation examples](https://scikit-learn.org/stable/modules/tree.html)

* **💻 Mini-Project Idea:** Predict passenger survival using random forests on the Titanic dataset and analyze feature importances.

#### Module 9: Clustering (K-Means, Hierarchical)

* **🧠 Key Concepts:**
  - K-Means: centroids, inertia, and assignment steps
  - Elbow method and silhouette score for choosing K
  - Hierarchical clustering: agglomerative vs. divisive, linkage criteria (single, complete)
  - Dendrograms for visualizing cluster hierarchies
  - Handling outliers and non-spherical clusters

* **🎯 Learning Objectives:** Apply unsupervised clustering to discover patterns in unlabeled data and evaluate cluster quality.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Unsupervised Learning, Recommenders, Reinforcement Learning by Andrew Ng (free audit, clustering section)](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning)
  * **Visual Intuition:** [StatQuest with Josh Starmer - K-Means Clustering and Hierarchical Clustering videos](https://www.youtube.com/watch?v=4b5d3muPQmA)
  * **Code Implementation:** [Scikit-learn Clustering documentation with examples](https://scikit-learn.org/stable/modules/clustering.html)

* **💻 Mini-Project Idea:** Segment customers into groups using K-Means on the Mall Customers dataset based on spending habits.

### Phase 3: Intermediate ML & The Art of Modeling

#### Module 10: Dimensionality Reduction (PCA & Others)

* **🧠 Key Concepts:**
  - Curse of dimensionality and variance preservation
  - PCA: covariance matrix, eigendecomposition, principal components
  - Explained variance ratio and component selection
  - Alternatives: t-SNE for visualization, LDA for supervised reduction
  - Applications in noise reduction and feature extraction

* **🎯 Learning Objectives:** Reduce high-dimensional data while retaining key information to improve model efficiency and visualization.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Khan Academy - Principal Component Analysis section](https://www.youtube.com/playlist?list=PLbPhAbAhvjUzeLkPVnv0kc3_9rAfXpGtS)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Principal Component Analysis (PCA) video](https://www.youtube.com/watch?v=FgakZw6K1QQ)
  * **Code Implementation:** [Scikit-learn Dimensionality Reduction documentation examples](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)

* **💻 Mini-Project Idea:** Apply PCA to compress and visualize handwritten digits from the MNIST dataset in 2D space.

#### Module 11: Gradient Boosting Machines (XGBoost, LightGBM)

* **🧠 Key Concepts:**
  - Boosting ensemble: sequential learning from residuals
  - Hyperparameters: learning rate, max depth, subsample
  - Regularization: L1/L2 penalties, early stopping
  - XGBoost features: built-in cross-validation, handling missing values
  - LightGBM advantages: faster training via histogram-based splitting

* **🎯 Learning Objectives:** Build and optimize gradient boosting models for superior performance on structured data tasks.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Kaggle - Intro to Gradient Boosting micro-course](https://www.kaggle.com/code/egazakharenko/gradient-boosting-from-scratch-full-tutorial)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Gradient Boost Part 1-4 series](https://www.youtube.com/watch?v=3CC4N4z3GJc)
  * **Code Implementation:** [XGBoost official documentation getting started guide](https://xgboost.readthedocs.io/)

* **💻 Mini-Project Idea:** Predict flight delays using XGBoost on the Airlines dataset and tune hyperparameters with grid search.

#### Module 12: Model Evaluation & Selection

* **🧠 Key Concepts:**
  - Data splitting: train-test split, k-fold cross-validation
  - Classification metrics: precision, recall, F1-score, ROC-AUC
  - Regression metrics: MSE, RMSE, MAPE
  - Confusion matrices and learning curves
  - Model selection: grid search, random search, Bayesian optimization

* **🎯 Learning Objectives:** Assess model performance using robust metrics and select the best model via systematic comparison.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Scikit-learn User Guide - Model Evaluation section](https://scikit-learn.org/stable/user_guide.html)
  * **Visual Intuition:** [StatQuest with Josh Starmer - ROC and AUC video](https://www.youtube.com/watch?v=4jRBRDbJemM&pp=0gcJCfwAo7VqN5tD)
  * **Code Implementation:** [Kaggle - Model Validation tutorial notebook](https://www.kaggle.com/code/dansbecker/model-validation)

* **💻 Mini-Project Idea:** Evaluate and compare classifiers (e.g., logistic regression vs. random forest) for credit card fraud detection on the Credit Card Fraud dataset.

#### Module 13: Feature Engineering

* **🧠 Key Concepts:**
  - Encoding: one-hot, label, target encoding
  - Scaling and normalization: StandardScaler, RobustScaler
  - Creating interactions: polynomial features, binning
  - Handling missing data: imputation, deletion strategies
  - Feature selection: filter methods (correlation), wrapper methods (RFE)

* **🎯 Learning Objectives:** Engineer features from raw data to enhance model accuracy and interpretability.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Kaggle - Feature Engineering micro-course](https://www.kaggle.com/learn/feature-engineering)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Feature Selection video](https://www.youtube.com/watch?v=wpNl-JwwplA)
  * **Code Implementation:** [Scikit-learn Preprocessing documentation examples](https://scikit-learn.org/stable/modules/preprocessing.html)

* **💻 Mini-Project Idea:** Enhance prediction accuracy for house prices by engineering features on the Ames Housing dataset.

#### Module 14: Comprehensive ML Project

* **🧠 Key Concepts:**
  - End-to-end pipeline: data ingestion, cleaning, EDA
  - Integrating feature engineering and model selection
  - Hyperparameter tuning with cross-validation
  - Model interpretation: SHAP values, partial dependence plots
  - Documentation and reproducibility best practices

* **🎯 Learning Objectives:** Develop a complete ML project from data to deployment-ready model, showcasing integrated skills.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Kaggle - Machine Learning Explainability micro-course](https://www.kaggle.com/learn/machine-learning-explainability)
  * **Visual Intuition:** [Abhishek Thakur - Approaching (Almost) Any Machine Learning Problem (free YouTube series)](https://www.youtube.com/watch?v=uWVR_axaVwk)
  * **Code Implementation:** [Towards Data Science - End-to-End Machine Learning Project tutorial](https://towardsdatascience.com/end-to-end-machine-learning-in-azure-1429528ecbe5/)

* **💻 Mini-Project Idea:** Build an end-to-end pipeline to predict diabetes outcomes using the Pima Indians Diabetes dataset, including EDA, feature engineering, and model comparison.

### Phase 4: Introduction to Deep Learning & Neural Networks

#### Module 15: Anatomy of a Neural Network

* **🧠 Key Concepts:**
  - Neurons, layers (input, hidden, output), weights, biases
  - Activation functions: sigmoid, tanh, ReLU, softmax
  - Forward propagation and matrix multiplications
  - Network architectures: feedforward, dense layers
  - Initialization techniques: Xavier, He initialization

* **🎯 Learning Objectives:** Describe and simulate the structure and data flow in a basic neural network.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Neural Networks and Deep Learning by Andrew Ng (free audit, Course 1)](https://www.coursera.org/learn/neural-networks-deep-learning)
  * **Visual Intuition:** [3Blue1Brown - Neural Networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  * **Code Implementation:** [Keras Sequential Model guide on TensorFlow documentation](https://www.tensorflow.org/guide/keras/sequential_model)

* **💻 Mini-Project Idea:** Construct a simple feedforward network in Keras to approximate a sine function using synthetic data.

#### Module 16: Backpropagation & Gradient Descent Variants

* **🧠 Key Concepts:**
  - Backpropagation: chain rule for error propagation
  - Loss functions: MSE for regression, cross-entropy for classification
  - Optimizers: SGD, momentum, Adam, RMSprop
  - Learning rate scheduling and adaptive learning
  - Vanishing/exploding gradients and mitigation

* **🎯 Learning Objectives:** Implement training loops with backpropagation and compare optimizer effects on convergence.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [fast.ai - Practical Deep Learning for Coders (free, Lesson 2 on backprop)](https://course.fast.ai/Lessons/part2.html)
  * **Visual Intuition:** [3Blue1Brown - Backpropagation calculus video](https://www.youtube.com/watch?v=tIeHLnjs5U8&vl=en)
  * **Code Implementation:** [PyTorch Autograd tutorial on official documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

* **💻 Mini-Project Idea:** Implement backpropagation from scratch in NumPy for a binary classification task on the Moons dataset.

#### Module 17: Building Your First Neural Network

* **🧠 Key Concepts:**
  - Framework basics: Keras layers, model compilation
  - Data preparation: splitting, normalization, batching
  - Training cycle: epochs, validation, callbacks
  - Model saving and loading
  - Basic hyperparameter tuning: layer sizes, activations

* **🎯 Learning Objectives:** Code and train a complete neural network using a high-level framework like Keras.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [TensorFlow - Keras for Beginners tutorial series](https://www.tensorflow.org/tutorials/quickstart/beginner)
  * **Visual Intuition:** [deeplizard - Keras Sequential API video series](https://www.youtube.com/watch?v=HrfrN3hn7QE)
  * **Code Implementation:** [Kaggle - Intro to Deep Learning notebook](https://www.kaggle.com/learn/intro-to-deep-learning)

* **💻 Mini-Project Idea:** Build and train a multi-layer perceptron in Keras for digit classification on the MNIST dataset.

#### Module 18: Regularization Techniques

* **🧠 Key Concepts:**
  - Overfitting detection: training vs. validation curves
  - L1/L2 regularization for weight penalties
  - Dropout: random neuron deactivation
  - Early stopping and data augmentation
  - Batch normalization for stable training

* **🎯 Learning Objectives:** Apply regularization methods to improve neural network generalization on unseen data.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Improving Deep Neural Networks by Andrew Ng (free audit)](https://www.coursera.org/learn/deep-neural-network)
  * **Visual Intuition:** [StatQuest with Josh Starmer - Regularization Part 1-3 series](https://www.youtube.com/watch?v=Q81RR3yKn30)
  * **Code Implementation:** [Keras Regularizers and Dropout documentation examples](https://keras.io/api/layers/regularization_layers/dropout/)

* **💻 Mini-Project Idea:** Add dropout and L2 regularization to a neural network for fashion item classification on the Fashion MNIST dataset to reduce overfitting.

### Phase 5: Specializations in Deep Learning

#### Module 19: Convolutional Neural Networks (CNNs)

* **🧠 Key Concepts:**
  - Convolution operations, filters, and feature maps
  - Pooling layers: max, average, global
  - Strides, padding, and receptive fields
  - Architectures: LeNet, AlexNet basics
  - Transfer learning with pre-trained models (e.g., ResNet)

* **🎯 Learning Objectives:** Design CNNs for image-related tasks, leveraging convolution for spatial hierarchies.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Stanford CS231n - Convolutional Neural Networks for Visual Recognition (free lecture notes)](https://cs231n.github.io/)
  * **Visual Intuition:** [Computerphile - Convolutional Neural Networks video](https://www.youtube.com/watch?v=py5byOOHZM8)
  * **Code Implementation:** [TensorFlow - CNN for Image Classification tutorial](https://www.tensorflow.org/tutorials/images/cnn)

* **💻 Mini-Project Idea:** Fine-tune a pre-trained CNN for cat vs. dog classification on the Cats vs. Dogs dataset.

#### Module 20: Recurrent Neural Networks (RNNs) & LSTMs

* **🧠 Key Concepts:**
  - Sequence processing: time steps, hidden states
  - Vanilla RNNs and backpropagation through time (BPTT)
  - LSTMs: gates (forget, input, output), cell state
  - GRUs as LSTM variants
  - Bidirectional RNNs and attention basics

* **🎯 Learning Objectives:** Model sequential data like text or time series using RNN variants.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Sequence Models by Andrew Ng (free audit)](https://www.coursera.org/learn/nlp-sequence-models)
  * **Visual Intuition:** [Illustrated Guide to LSTMs and GRUs by Michael Phi (blog post)](https://medium.com/data-science/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
  * **Code Implementation:** [Keras RNN and LSTM layers documentation examples](https://keras.io/api/layers/recurrent_layers/lstm/)

* **💻 Mini-Project Idea:** Train an LSTM for sentiment analysis on the IMDB movie reviews dataset.

#### Module 21: Transformers Basics

* **🧠 Key Concepts:**
  - Self-attention mechanism and multi-head attention
  - Positional encodings for sequence order
  - Encoder-decoder structure
  - Feed-forward networks and layer normalization
  - Pre-trained models: BERT for understanding, GPT for generation

* **🎯 Learning Objectives:** Understand and apply transformers for NLP tasks like classification and generation.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [The Illustrated Transformer by Jay Alammar (free blog)](https://jalammar.github.io/illustrated-transformer/)
  * **Visual Intuition:** [Jalammar - Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
  * **Code Implementation:** [Hugging Face Transformers quickstart tutorial](https://huggingface.co/docs/transformers/en/quicktour)

* **💻 Mini-Project Idea:** Fine-tune BERT for text classification on the GLUE benchmark's SST-2 dataset.

### Phase 6: Becoming a Practitioner - MLOps and Beyond

#### Module 22: Model Deployment

* **🧠 Key Concepts:**
  - Web frameworks: Flask vs. FastAPI for APIs
  - Model serialization: pickle, ONNX
  - Endpoint creation: inference requests, responses
  - Containerization basics with Docker
  - Scaling: load balancing, cloud deployment

* **🎯 Learning Objectives:** Deploy trained models as accessible web services for real-time predictions.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [freeCodeCamp - Deploy Machine Learning Models with Flask](https://www.freecodecamp.org/news/deploy-your-machine-learning-models-for-free/)
  * **Visual Intuition:** [AssemblyAI - FastAPI for ML Deployment video](https://www.youtube.com/watch?v=h5wLuVDr0oc&pp=0gcJCfwAo7VqN5tD)
  * **Code Implementation:** [FastAPI official tutorial for ML models](https://fastapi.tiangolo.com/tutorial/)

* **💻 Mini-Project Idea:** Deploy a sentiment analysis model as a REST API using FastAPI on the IMDB dataset.

#### Module 23: Version Control for ML

* **🧠 Key Concepts:**
  - Git workflows: branching, merging, pull requests
  - Data versioning with DVC: tracking large files
  - Experiment tracking: MLflow basics
  - Reproducibility: seeds, environments (virtualenv)
  - Collaboration: GitHub for ML projects

* **🎯 Learning Objectives:** Manage ML code, data, and experiments to ensure reproducibility and collaboration.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [GitHub Docs - Git and GitHub Learning Lab](https://docs.github.com/en/get-started/start-your-journey/git-and-github-learning-resources)
  * **Visual Intuition:** [Corey Schafer - Git Tutorial for Beginners series](https://www.youtube.com/watch?v=HVsySz-h9r4)
  * **Code Implementation:** [DVC official getting started guide](https://dvc.org/doc/start)

* **💻 Mini-Project Idea:** Use Git and DVC to version control an image classification project on the CIFAR-10 dataset.

#### Module 24: Cloud AI Platforms

* **🧠 Key Concepts:**
  - AWS SageMaker: notebooks, training jobs, endpoints
  - Google Cloud Vertex AI: pipelines, AutoML
  - Azure ML: workspaces, experiments, deployments
  - Cost management: spot instances, free tiers
  - Integration: APIs, SDKs for model hosting

* **🎯 Learning Objectives:** Utilize cloud services to scale ML training and deployment beyond local resources.

* **📚 Key Resources (Free & Online):**
  * **Primary Course/Text:** [Coursera - Google Cloud Machine Learning Engineering (free audit)](https://www.coursera.org/professional-certificates/preparing-for-google-cloud-machine-learning-engineer-professional-certificate)
  * **Visual Intuition:** [AWS - SageMaker Overview video series](https://www.youtube.com/watch?v=mzkHGEyAPEw)
  * **Code Implementation:** [AWS SageMaker Python SDK tutorial](https://sagemaker.readthedocs.io/en/stable/overview.html)

* **💻 Mini-Project Idea:** Train and deploy a regression model for housing prices on Google Cloud Vertex AI using the Boston Housing dataset.
