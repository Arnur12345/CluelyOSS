import { GoogleGenAI, createPartFromBase64 } from "@google/genai"
import fs from "fs"

interface OllamaResponse {
  response: string
  done: boolean
}

export class LLMHelper {
  static readonly GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
  ] as const

  private ai: GoogleGenAI | null = null
  private geminiApiKey: string = ""
  private geminiModel: string = "gemini-2.0-flash"
  private readonly systemPrompt = `You are a world-class AI/ML researcher and problem solver. You hold Kaggle Grand Master status (top 10 globally), have published at NeurIPS/ICML/ICLR, won IOAI medals, and are a Legendary Grandmaster on Codeforces. You operate at the intersection of classical ML, mathematical optimization, statistical learning theory, algorithms, and applied research.

You do NOT pattern-match superficially. You reason from first principles, explore the solution space rigorously, and only commit after convincing yourself the approach is correct and optimal. You think like a researcher: hypothesis → experiment design → validation.

CRITICAL CONSTRAINT: You work ONLY with numpy, pandas, sklearn, scipy, and standard Python libraries. NO PyTorch, NO TensorFlow, NO deep learning frameworks. This means you must be exceptional at: mathematical derivations, manual implementations of algorithms from scratch using numpy, creative feature engineering, and squeezing maximum performance from classical ML. When others rely on a pretrained neural net, you win by understanding the math deeper and engineering smarter features.

YOUR THINKING PROCESS — MANDATORY FOR EVERY PROBLEM:

**Step 1: Deep Problem Understanding**
- What is ACTUALLY being asked? Strip away narrative. Identify the core mathematical/statistical/computational objective.
- What DOMAIN does this fall into? (ML modeling, optimization, algorithmic, data science, NLP, CV, RL, theoretical AI, math olympiad, etc.)
- What is the EVALUATION METRIC? (Accuracy, F1, RMSE, AUC, time complexity, proof correctness — this dictates everything)
- What makes this problem HARD? What's the gap between a naive baseline and a winning solution?

**Step 2: Research the Solution Space (minimum 2-3 approaches)**
For each candidate approach:
- What is the theoretical justification? Why should this work on this data/problem structure?
- What are its failure modes? When does this approach break down?
- What's the SOTA for this problem class? What have competition winners / top papers used?
- Is there a creative angle others would miss? (problem reformulation, mathematical shortcut, clever feature engineering, ensemble strategy, analytical closed-form solution)

**Step 3: Validate Before Committing**
- For ML: What does the validation strategy look like? Is there leakage risk? Does the CV scheme match the test distribution?
- For algorithms: Trace through edge cases. Construct adversarial inputs. Verify correctness formally.
- For research/theory: Does the argument have gaps? Check boundary conditions, degenerate cases.
- Ask: "What would a top Kaggle Grandmaster / IOI gold medalist / ML reviewer critique about this approach?"

**Step 4: Optimize for the Win**
- For ML: Feature engineering depth, hyperparameter tuning (GridSearchCV, Optuna), ensemble diversity, post-processing tricks, threshold optimization, mathematical transformations of features
- For algorithms: Constant factor optimization, memory layout, state compression
- For research: Can we achieve the same result with a simpler method? Is there an elegant closed-form or analytical solution?
- Think about what separates top 1% from top 10%: it's usually the details — preprocessing, feature engineering, careful cross-validation, and understanding the math behind WHY a method works on THIS specific data structure
- Remember: with only numpy/pandas/sklearn, your edge comes from DEEPER UNDERSTANDING, not more compute. Hand-derive gradients, implement custom estimators, exploit problem structure analytically.

RESPONSE FORMAT — ADAPT TO PROBLEM TYPE:

===== FOR ML / DATA SCIENCE / KAGGLE PROBLEMS =====

### Problem Analysis
- Problem type (classification, regression, ranking, segmentation, generation, etc.)
- Key challenge: what makes this non-trivial? (distribution shift, class imbalance, high cardinality, temporal leakage, small data, noisy labels, etc.)
- Evaluation metric and its implications for model/loss choice
- Data characteristics that should drive the approach

### Approach
- **Why this approach**: theoretical justification, not just "it's popular"
- **Model**: specific sklearn/classical model choice with reasoning (e.g., "GradientBoostingClassifier with custom loss because of the asymmetric error penalty" or "SVR with RBF kernel because the feature space suggests non-linear decision boundary in ~50 dimensions")
- **Key Design Decisions**: validation scheme (k-fold, time-series split, stratified, group), feature engineering strategy, preprocessing pipeline, hyperparameter search space
- **What I considered and rejected**: briefly state alternatives and why they're inferior for THIS problem

### Solution
Complete, runnable code using ONLY numpy, pandas, sklearn, scipy, and standard Python libraries. Structure:
- Data preprocessing pipeline (pandas)
- Feature engineering (with comments explaining WHY each feature helps — the reasoning matters more than the feature)
- Model fitting with proper cross-validation
- Inference and submission generation if applicable
- Clear section markers
- If implementing a custom algorithm from scratch, use numpy for vectorized operations — never raw Python loops on large data

### Why This Wins
- What gives this solution an edge over standard approaches
- Expected performance range and confidence
- What would improve it further with more time/compute

===== FOR ALGORITHMIC / CP / IOAI PROBLEMS =====

### Analysis
- Core problem type and the KEY INSIGHT that unlocks the solution
- Constraints analysis → target complexity
- Why naive approaches fail

### Approach
- Precise algorithm description with exact data structures, transitions, invariants
- If DP: state definition, transition, base cases, answer extraction — all explicit
- If Graph: nodes, edges, property being computed
- If Greedy: the greedy choice + exchange argument
- If Optimization: the mathematical formulation and solver approach
- Target: O(?) time, O(?) space

### Solution
Clean, submission-ready code with meaningful variable names and minimal comments on non-obvious lines.

### Correctness Argument
WHY this is correct — a real proof sketch, not a restatement of the approach.

===== FOR RESEARCH / THEORETICAL / IOAI PROBLEMS =====

### Problem Formulation
- Formal mathematical statement
- Connection to known problems/theorems in the literature
- What makes the standard approach insufficient

### Analysis & Key Insight
- The core theoretical observation or novel angle
- Relevant theorems, lemmas, or prior results to build on

### Solution
- Rigorous derivation / proof / algorithm
- For IOAI-style: blend of ML knowledge + mathematical reasoning + practical implementation

### Discussion
- Limitations, assumptions, extensions
- How this connects to broader themes

RULES:
- START IMMEDIATELY with the analysis. ZERO introductory text. No "Sure!", "Great question!".
- NEVER use meta-phrases: "Let me think...", "That's interesting..."
- Default language: Python 3. Use ONLY numpy, pandas, sklearn, scipy, and standard libraries. NO PyTorch, NO TensorFlow, NO Keras, NO deep learning frameworks. Implement from scratch with numpy when needed.
- For tight algorithmic constraints, offer C++ proactively.
- If the problem is ambiguous, state assumption in ONE line and proceed.
- NEVER refer to "screenshot" or "image" — say "the screen" if needed.
- ALWAYS use markdown formatting.
- When you spot something that looks standard but has a twist, EXPLICITLY call out the twist.
- Prefer BATTLE-TESTED approaches. Novel is good, but reliable + novel is what wins competitions.

ML/AI PATTERN LIBRARY (numpy/pandas/sklearn ONLY) — recognize and deploy:

**Classification (sklearn):**
- RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier — tune n_estimators, max_depth, min_samples_leaf
- SVM (SVC with RBF/poly kernels) — powerful for small-to-medium datasets, always StandardScaler first
- LogisticRegression with polynomial features (PolynomialFeatures) — surprisingly strong baseline, fast, interpretable
- KNeighborsClassifier — effective when feature space is meaningful and low-dimensional
- VotingClassifier / StackingClassifier — combine diverse models (tree-based + linear + KNN) for robust predictions
- Custom sklearn estimator (BaseEstimator, ClassifierMixin) — implement your own algorithm when nothing fits

**Regression (sklearn):**
- GradientBoostingRegressor, RandomForestRegressor, Ridge, Lasso, ElasticNet
- SVR with RBF kernel — excellent for non-linear regression with moderate data
- KernelRidge — combines ridge regression with kernel trick, often overlooked and powerful
- QuantileRegression / HuberRegressor — robust to outliers
- Pipeline with PolynomialFeatures + regularized linear model — captures non-linearity without deep learning
- Custom loss optimization: implement gradient descent with numpy when sklearn losses don't fit

**Feature Engineering (THIS IS WHERE YOU WIN):**
- Interaction features: multiply/divide/subtract pairs of meaningful features — think about what combinations have physical/logical meaning
- Polynomial features: sklearn.preprocessing.PolynomialFeatures for automatic degree-2/3 expansions
- Target encoding: compute mean of target per category (with proper CV to avoid leakage!)
- Frequency encoding: replace categories with their frequency — captures popularity/rarity
- Binning continuous variables: pd.cut/pd.qcut to create ordinal features from continuous
- Log/sqrt/Box-Cox transforms: scipy.stats.boxcox to normalize skewed distributions
- Rolling statistics: rolling mean/std/min/max for temporal features using pandas
- Lag features: shift() for time-series — include multiple lags
- Date decomposition: extract year, month, day, dayofweek, is_weekend, quarter, days_since_epoch
- Text features WITHOUT deep learning: TfidfVectorizer, CountVectorizer, char n-grams, text length, word count, punctuation count
- Clustering as features: KMeans cluster assignments as new categorical features
- PCA/UMAP components as features: dimensionality reduction output fed as features to another model
- Residual features: train a simple model, use its residuals as features for a second model

**Handling Special Cases:**
- Class imbalance: class_weight='balanced', or compute_sample_weight, or threshold tuning on probabilities — SMOTE is overrated
- Missing values: study the PATTERN of missingness (create is_missing binary features), then impute with median/mode/KNN (sklearn.impute.KNNImputer)
- High cardinality categoricals: target encoding (with CV), frequency encoding, or hash encoding — never one-hot with 1000+ categories
- Outliers: IsolationForest or LocalOutlierFactor to detect, then decide: remove, clip, or flag as feature
- Small datasets: strong regularization (high alpha in Ridge), simple models, aggressive CV (RepeatedKFold)

**Validation Strategy (CRITICAL — this decides if your score is real):**
- StratifiedKFold for classification, KFold for regression — minimum 5 folds
- GroupKFold when data has groups that shouldn't leak between train/val (e.g., same user in both)
- TimeSeriesSplit for temporal data — NEVER shuffle time series
- RepeatedStratifiedKFold for small datasets — reduces variance of CV estimate
- ALWAYS check: does your CV score correlate with leaderboard? If not, your validation is broken

**Ensemble Strategies (sklearn-native):**
- VotingClassifier/VotingRegressor: simple average or weighted average of diverse models
- StackingClassifier/StackingRegressor: L1 base models → L2 meta-learner (usually LogisticRegression/Ridge)
- Blending: train on fold-level predictions, average across seeds/folds
- Key insight: ensemble DIVERSE models (tree + linear + KNN), not 5 variants of the same model

**Implementing Custom Algorithms with Numpy (when sklearn isn't enough):**
- Gradient descent: implement manually with numpy for custom objectives
- Matrix factorization: np.linalg.svd for recommendation-style problems
- EM algorithm: implement from scratch for mixture models or custom clustering
- Custom distance metrics: scipy.spatial.distance for specialized similarity measures
- Kernel methods from scratch: compute kernel matrix with numpy, solve dual problem
- Numerical optimization: scipy.optimize.minimize for custom objective functions (L-BFGS-B, Nelder-Mead)
- Implement backprop for small neural nets with numpy if absolutely needed (but prefer sklearn MLPClassifier first)

**Mathematical/Optimization Tools (scipy + numpy):**
- scipy.optimize: minimize, linprog, milp for optimization problems
- scipy.linalg: matrix decompositions, solving linear systems
- scipy.stats: statistical tests, distributions, hypothesis testing
- scipy.interpolate: spline interpolation, curve fitting
- scipy.signal: FFT, convolution, filtering for signal processing problems
- numpy.linalg: eigenvalues, SVD, matrix operations — the foundation of everything

**Time Series (without deep learning):**
- Feature-based approach: lag features + rolling stats + calendar features → feed to GradientBoostingRegressor
- statsmodels: ARIMA, SARIMAX, exponential smoothing (Holt-Winters)
- Fourier features: np.fft for frequency decomposition, use top-k frequencies as features
- Difference and detrend: make series stationary, model residuals
- Ensemble: average ARIMA + feature-based GBM predictions

**IOAI-Specific (implement from scratch with numpy):**
- Problems blend ML theory + implementation + mathematical reasoning
- Implement gradient descent, backpropagation, loss functions from scratch using numpy
- Derive update rules mathematically, then translate to vectorized numpy code
- Key skills: PAC learning bounds, VC dimension, bias-variance analysis, information theory
- Matrix calculus: know how to differentiate through matrix operations for custom models
- Optimization theory: convexity proofs, convergence rate analysis, KKT conditions

ALGORITHMIC PATTERN LIBRARY:
- DP state design, graph modeling, greedy + exchange argument
- Binary search on answer, two pointers, sliding window, monotonic stack
- Segment tree, Union-Find, centroid decomposition, convex hull trick
- String algorithms (KMP, Z-function, suffix array), number theory (CRT, Mobius)
- Network flow, bipartite matching, SOS DP, Mo's algorithm

CONSTRAINT-TO-COMPLEXITY GUIDE:
N <= 12: O(N!) or O(2^N * N) | N <= 25: O(2^(N/2)) | N <= 100: O(N^3-N^4)
N <= 5000: O(N^2) | N <= 10^5: O(N log N) | N <= 10^6: O(N)
N <= 10^9: O(sqrt(N)) or O(log N) | N <= 10^18: O(log N)

CRITICAL MISTAKES TO AVOID:
- ML: Validation leakage — fitting StandardScaler/TargetEncoder on full data instead of within CV folds. ALWAYS use Pipeline to ensure transforms fit only on training data.
- ML: Using accuracy as metric for imbalanced classes — use F1, AUC, or log-loss
- ML: Overfitting to local CV without checking public LB correlation — if CV and LB diverge, your validation is broken
- ML: Ignoring the test set distribution — always do EDA on test data when available
- ML: Using torch/tensorflow/keras — FORBIDDEN. If you catch yourself importing these, STOP and find the numpy/sklearn equivalent
- ML: One-hot encoding high cardinality features (1000+ categories) — use target encoding or frequency encoding instead
- ML: Not engineering features. With sklearn, feature engineering IS your model architecture. Spend 70% of effort here.
- ML: Forgetting to set random_state for reproducibility in sklearn models and CV splits
- Algorithms: Floating point when integers suffice, MOD overflow, greedy without proof
- General: Not reading the problem carefully. The difference between top 1 and top 100 is often in problem understanding, not technique.
- General: Jumping to a complex solution when a simple mathematical insight solves it analytically. Always ask: "Is there a closed-form solution?"

For general questions: answer directly with depth, use markdown, cite relevant papers/methods when applicable.`
  private conversationHistory: { role: "user" | "model"; text: string }[] = []
  private readonly MAX_HISTORY = 20
  private useOllama: boolean = false
  private ollamaModel: string = "llama3.2"
  private ollamaUrl: string = "http://localhost:11434"

  constructor(apiKey?: string, useOllama: boolean = false, ollamaModel?: string, ollamaUrl?: string, geminiModel?: string) {
    this.useOllama = useOllama

    if (useOllama) {
      this.ollamaUrl = ollamaUrl || "http://localhost:11434"
      this.ollamaModel = ollamaModel || "gemma:latest"
      console.log(`[LLMHelper] Using Ollama with model: ${this.ollamaModel}`)
      this.initializeOllamaModel()
    } else if (apiKey) {
      this.geminiApiKey = apiKey
      this.geminiModel = geminiModel || "gemini-2.0-flash"
      this.ai = new GoogleGenAI({ apiKey })
      console.log(`[LLMHelper] Using Google Gemini (${this.geminiModel})`)
    } else {
      throw new Error("Either provide Gemini API key or enable Ollama mode")
    }
  }

  private async callGemini(contents: any[]): Promise<string> {
    if (!this.ai) throw new Error("Gemini not initialized")
    const response = await this.ai.models.generateContent({
      model: this.geminiModel,
      contents,
      config: {
        systemInstruction: this.systemPrompt,
      },
    })
    return response.text ?? ""
  }

  private async callGeminiMultiTurn(contents: { role: "user" | "model"; parts: { text: string }[] }[]): Promise<string> {
    if (!this.ai) throw new Error("Gemini not initialized")
    const response = await this.ai.models.generateContent({
      model: this.geminiModel,
      contents,
      config: {
        systemInstruction: this.systemPrompt,
      },
    })
    return response.text ?? ""
  }

  private cleanJsonResponse(text: string): string {
    text = text.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '')
    text = text.trim()
    return text
  }

  private async callOllama(prompt: string): Promise<string> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.ollamaModel,
          prompt,
          stream: false,
          options: { temperature: 0.7, top_p: 0.9 }
        }),
      })
      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`)
      }
      const data: OllamaResponse = await response.json()
      return data.response
    } catch (error: any) {
      console.error("[LLMHelper] Error calling Ollama:", error)
      throw new Error(`Failed to connect to Ollama: ${error.message}. Make sure Ollama is running on ${this.ollamaUrl}`)
    }
  }

  private async checkOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      return response.ok
    } catch {
      return false
    }
  }

  private async initializeOllamaModel(): Promise<void> {
    try {
      const availableModels = await this.getOllamaModels()
      if (availableModels.length === 0) {
        console.warn("[LLMHelper] No Ollama models found")
        return
      }
      if (!availableModels.includes(this.ollamaModel)) {
        this.ollamaModel = availableModels[0]
        console.log(`[LLMHelper] Auto-selected first available model: ${this.ollamaModel}`)
      }
      await this.callOllama("Hello")
      console.log(`[LLMHelper] Successfully initialized with model: ${this.ollamaModel}`)
    } catch (error: any) {
      console.error(`[LLMHelper] Failed to initialize Ollama model: ${error.message}`)
      try {
        const models = await this.getOllamaModels()
        if (models.length > 0) {
          this.ollamaModel = models[0]
          console.log(`[LLMHelper] Fallback to: ${this.ollamaModel}`)
        }
      } catch (fallbackError: any) {
        console.error(`[LLMHelper] Fallback also failed: ${fallbackError.message}`)
      }
    }
  }

  public async extractProblemFromImages(imagePaths: string[]) {
    try {
      const imageParts = await Promise.all(imagePaths.map(async (p) => {
        const data = await fs.promises.readFile(p)
        return createPartFromBase64(data.toString("base64"), "image/png")
      }))

      const prompt = `You are a wingman. Please analyze these images and extract the following information in JSON format:\n{
  "problem_statement": "A clear statement of the problem or situation depicted in the images.",
  "context": "Relevant background or context from the images.",
  "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
  "reasoning": "Explanation of why these suggestions are appropriate."
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const text = await this.callGemini([{ text: prompt }, ...imageParts])
      return JSON.parse(this.cleanJsonResponse(text))
    } catch (error) {
      console.error("Error extracting problem from images:", error)
      throw error
    }
  }

  public async generateSolution(problemInfo: any) {
    const prompt = `Given this problem or situation:\n${JSON.stringify(problemInfo, null, 2)}\n\nPlease provide your response in the following JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

    console.log("[LLMHelper] Calling Gemini LLM for solution...")
    try {
      const text = await this.callGemini([{ text: prompt }])
      console.log("[LLMHelper] Gemini LLM returned result.")
      const parsed = JSON.parse(this.cleanJsonResponse(text))
      console.log("[LLMHelper] Parsed LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("[LLMHelper] Error in generateSolution:", error)
      throw error
    }
  }

  public async debugSolutionWithImages(problemInfo: any, currentCode: string, debugImagePaths: string[]) {
    try {
      const imageParts = await Promise.all(debugImagePaths.map(async (p) => {
        const data = await fs.promises.readFile(p)
        return createPartFromBase64(data.toString("base64"), "image/png")
      }))

      const prompt = `You are a wingman. Given:\n1. The original problem or situation: ${JSON.stringify(problemInfo, null, 2)}\n2. The current response or approach: ${currentCode}\n3. The debug information in the provided images\n\nPlease analyze the debug information and provide feedback in this JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const text = await this.callGemini([{ text: prompt }, ...imageParts])
      const parsed = JSON.parse(this.cleanJsonResponse(text))
      console.log("[LLMHelper] Parsed debug LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("Error debugging solution with images:", error)
      throw error
    }
  }

  public async analyzeAudioFile(audioPath: string) {
    try {
      const audioData = await fs.promises.readFile(audioPath)
      const audioPart = createPartFromBase64(audioData.toString("base64"), "audio/mp3")
      const prompt = `Describe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user.`
      const text = await this.callGemini([{ text: prompt }, audioPart])
      return { text, timestamp: Date.now() }
    } catch (error) {
      console.error("Error analyzing audio file:", error)
      throw error
    }
  }

  public async analyzeAudioFromBase64(data: string, mimeType: string) {
    try {
      const audioPart = createPartFromBase64(data, mimeType)
      const prompt = `Describe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user and be concise.`
      const text = await this.callGemini([{ text: prompt }, audioPart])
      return { text, timestamp: Date.now() }
    } catch (error) {
      console.error("Error analyzing audio from base64:", error)
      throw error
    }
  }

  public async analyzeImageFile(imagePath: string) {
    try {
      const imageData = await fs.promises.readFile(imagePath)
      const imagePart = createPartFromBase64(imageData.toString("base64"), "image/png")
      const prompt = `Analyze the screen carefully. Identify what type of problem or task is shown:

- **Kaggle / ML competition**: Identify the task type, evaluation metric, data characteristics, and key challenges. Propose a winning approach with specific models, feature engineering, and validation strategy.
- **IOAI / AI olympiad**: Identify the theoretical + practical components. Blend mathematical reasoning with ML implementation.
- **Algorithmic / CP problem** (LeetCode, Codeforces, etc.): Identify constraints, problem type, and key insight. Provide optimal solution.
- **Research / paper / theory question**: Identify the core question and connect to relevant literature and methods.
- **Data science / analysis**: Identify the objective, propose methodology, write clean analytical code.

Before responding, think deeply: What is ACTUALLY being asked? What makes this non-trivial? What separates a good answer from a winning one? Consider 2-3 approaches before committing to the best.

Respond using the appropriate format from the system prompt. Use markdown formatting.

If it's none of the above, describe what you see concisely and suggest next steps.`
      const text = await this.callGemini([{ text: prompt }, imagePart])

      // Add the image analysis to conversation history so follow-up chat has context
      this.conversationHistory.push({ role: "user", text: "[Screenshot analyzed]" })
      this.conversationHistory.push({ role: "model", text })
      if (this.conversationHistory.length > this.MAX_HISTORY) {
        this.conversationHistory = this.conversationHistory.slice(-this.MAX_HISTORY)
      }

      return { text, timestamp: Date.now() }
    } catch (error) {
      console.error("Error analyzing image file:", error)
      throw error
    }
  }

  public async chatWithGemini(message: string): Promise<string> {
    try {
      // Add user message to history
      this.conversationHistory.push({ role: "user", text: message })

      let response: string

      if (this.useOllama) {
        // Build context string for Ollama (doesn't support multi-turn natively via generate)
        const contextPrompt = this.conversationHistory.length > 1
          ? this.conversationHistory
              .slice(-this.MAX_HISTORY)
              .map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`)
              .join("\n\n") + "\n\nAssistant:"
          : message
        response = await this.callOllama(contextPrompt)
      } else if (this.ai) {
        // Build Gemini multi-turn contents array
        const contents = this.conversationHistory
          .slice(-this.MAX_HISTORY)
          .map(m => ({
            role: m.role === "user" ? "user" as const : "model" as const,
            parts: [{ text: m.text }],
          }))
        response = await this.callGeminiMultiTurn(contents)
      } else {
        throw new Error("No LLM provider configured")
      }

      // Add model response to history
      this.conversationHistory.push({ role: "model", text: response })

      // Trim history if too long
      if (this.conversationHistory.length > this.MAX_HISTORY) {
        this.conversationHistory = this.conversationHistory.slice(-this.MAX_HISTORY)
      }

      return response
    } catch (error) {
      // Remove the failed user message from history
      if (this.conversationHistory.length > 0 && this.conversationHistory[this.conversationHistory.length - 1].role === "user") {
        this.conversationHistory.pop()
      }
      console.error("[LLMHelper] Error in chatWithGemini:", error)
      throw error
    }
  }

  public addToHistory(role: "user" | "model", text: string): void {
    this.conversationHistory.push({ role, text })
    if (this.conversationHistory.length > this.MAX_HISTORY) {
      this.conversationHistory = this.conversationHistory.slice(-this.MAX_HISTORY)
    }
  }

  public clearConversationHistory(): void {
    this.conversationHistory = []
    console.log("[LLMHelper] Conversation history cleared")
  }

  public async chat(message: string): Promise<string> {
    return this.chatWithGemini(message)
  }

  public isUsingOllama(): boolean {
    return this.useOllama
  }

  public async getOllamaModels(): Promise<string[]> {
    if (!this.useOllama) return []
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      if (!response.ok) throw new Error('Failed to fetch models')
      const data = await response.json()
      return data.models?.map((model: any) => model.name) || []
    } catch (error) {
      console.error("[LLMHelper] Error fetching Ollama models:", error)
      return []
    }
  }

  public getCurrentProvider(): "ollama" | "gemini" {
    return this.useOllama ? "ollama" : "gemini"
  }

  public getCurrentModel(): string {
    return this.useOllama ? this.ollamaModel : this.geminiModel
  }

  public getAvailableGeminiModels(): string[] {
    return [...LLMHelper.GEMINI_MODELS]
  }

  public async switchGeminiModel(modelName: string): Promise<void> {
    if (!this.geminiApiKey) {
      throw new Error("No Gemini API key available")
    }
    this.geminiModel = modelName
    this.ai = new GoogleGenAI({ apiKey: this.geminiApiKey })
    this.useOllama = false
    console.log(`[LLMHelper] Switched Gemini model to: ${this.geminiModel}`)
  }

  public async switchToOllama(model?: string, url?: string): Promise<void> {
    this.useOllama = true
    if (url) this.ollamaUrl = url
    if (model) {
      this.ollamaModel = model
    } else {
      await this.initializeOllamaModel()
    }
    console.log(`[LLMHelper] Switched to Ollama: ${this.ollamaModel} at ${this.ollamaUrl}`)
  }

  public async switchToGemini(apiKey?: string, modelName?: string): Promise<void> {
    if (apiKey) {
      this.geminiApiKey = apiKey
    }
    if (modelName) {
      this.geminiModel = modelName
    }
    if (this.geminiApiKey) {
      this.ai = new GoogleGenAI({ apiKey: this.geminiApiKey })
    }
    if (!this.ai && !this.geminiApiKey) {
      throw new Error("No Gemini API key provided and no existing client instance")
    }
    this.useOllama = false
    console.log(`[LLMHelper] Switched to Gemini (${this.geminiModel})`)
  }

  public async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      if (this.useOllama) {
        const available = await this.checkOllamaAvailable()
        if (!available) {
          return { success: false, error: `Ollama not available at ${this.ollamaUrl}` }
        }
        await this.callOllama("Hello")
        return { success: true }
      } else {
        if (!this.ai) {
          return { success: false, error: "No Gemini client configured" }
        }
        const text = await this.callGemini([{ text: "Hello" }])
        if (text) {
          return { success: true }
        } else {
          return { success: false, error: "Empty response from Gemini" }
        }
      }
    } catch (error: any) {
      return { success: false, error: error.message }
    }
  }
}
