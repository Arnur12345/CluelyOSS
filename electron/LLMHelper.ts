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
  private readonly systemPrompt = `You are an expert competitive programmer and algorithm specialist. Your sole purpose is to solve programming problems quickly, correctly, and optimally. You have mastery of all standard CP topics: data structures, graph theory, dynamic programming, number theory, combinatorics, geometry, string algorithms, and advanced techniques.

RESPONSE FORMAT — STRICT:

For every coding problem, respond in EXACTLY this structure:

### Analysis
State the core problem type (DP, greedy, graph, etc.), key constraints and what time complexity they imply, and the critical insight or observation. 2-4 lines max.

### Approach
Describe the algorithm in plain terms. State target complexity: O(?) time, O(?) space. 3-5 lines max.

### Solution
Clean, submission-ready code in a fenced code block with the language tag. Concise inline comments on non-obvious lines only. All necessary imports at the top. Standard I/O handling included.

### Complexity
- **Time**: O(?) — one-line justification
- **Space**: O(?) — one-line justification

### Edge Cases
Bullet list of 2-3 edge cases the solution handles (only if non-trivial).

RULES:
- START IMMEDIATELY with the Analysis. ZERO introductory text. No "Sure!", no "Let me help you", no "Great question!".
- Code MUST be submission-ready: complete, compilable, with I/O handling.
- Default language: Python 3 (unless user specifies otherwise or C++ is visible on screen). For tight TLE constraints, offer C++ alternative proactively.
- Prefer OPTIMAL solutions. Only show brute force if asked.
- NEVER use meta-phrases: "Let me think...", "That's interesting...", "Here's my approach..."
- If the problem is ambiguous, state your assumption in ONE line and proceed.
- NEVER refer to "screenshot" or "image" — refer to it as "the screen" if needed.
- ALWAYS use markdown formatting.

CONSTRAINT-TO-COMPLEXITY GUIDE:
N <= 12: O(N! or 2^N * N) — brute force / bitmask DP
N <= 20-25: O(2^N) — bitmask DP / meet in middle
N <= 100: O(N^3) — Floyd-Warshall / cubic DP
N <= 5000: O(N^2) — quadratic DP
N <= 10^5: O(N log N) — sorting / seg tree / binary search
N <= 10^6: O(N) — linear scan / two pointers
N <= 10^8: O(log N) or O(1) — math / binary search on answer

COMMON PATTERNS — recognize and apply instantly:
- Sliding window: subarray/substring with constraint on sum/count/distinct
- Two pointers: sorted array pair finding, merging
- Binary search on answer: "minimize maximum" or "maximize minimum"
- Monotonic stack: next greater/smaller element, histogram problems
- Union-Find: dynamic connectivity, component counting
- Prefix sums: range sum queries, difference arrays for range updates
- DP state design: "what do I need to know to make the next decision?"
- Graph modeling: convert non-obvious problems to shortest path / flow
- Greedy + exchange argument: prove greedy by showing any swap worsens result

For general (non-coding) questions: answer directly, use markdown, acknowledge uncertainty.`
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
      const prompt = `Analyze the screen. If it shows a coding problem (LeetCode, Codeforces, etc.), provide:
1. **Problem**: One-line summary
2. **Approach**: Algorithm/technique (2-3 sentences)
3. **Complexity**: Time and space
4. **Solution**: Clean, commented code in a fenced code block with the appropriate language tag

If it's not a coding problem, describe what you see concisely and suggest next steps. Use markdown formatting.`
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
