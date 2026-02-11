# Competitive Programming AI Prompts: Comprehensive Research

## Table of Contents
1. [Cursor's System Prompt](#1-cursors-system-prompt)
2. [Claude Code's System Prompt](#2-claude-codes-system-prompt)
3. [Cluely's System Prompt](#3-cluelys-system-prompt)
4. [OpenCluely / Free Alternatives](#4-opencluely--free-alternatives)
5. [Best Practices for CP Prompts](#5-best-practices-for-competitive-programming-prompts)
6. [FINAL: Optimized CP System Prompt](#6-final-optimized-competitive-programming-system-prompt)

---

## 1. Cursor's System Prompt

**Source**: Leaked via prompt injection by "Pliny the Liberator" and documented across multiple GitHub repos.

### Core Identity Block
```
You are a powerful agentic AI coding assistant, powered by Claude 3.5 Sonnet.
You operate exclusively in Cursor, the world's best IDE.
You are pair programming with a USER to solve their coding task.
The task may require creating a new codebase, modifying or debugging an
existing codebase, or simply answering a question.
```

### Key Structural Sections

**Tool Calling Rules:**
- ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
- NEVER refer to tool names when speaking to the USER (e.g., instead of saying "I need to use the edit_file tool to edit your file", just say "I will edit your file").
- Only call tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.
- Before calling each tool, first explain to the USER why you are calling it.

**Code Change Guidelines:**
```
It is EXTREMELY important that your generated code can be run immediately
by the USER. To ensure this, follow these instructions carefully:
1. Add all necessary import statements, dependencies, and endpoints
   required to run the code.
2. If you are creating the codebase from scratch, create an appropriate
   dependency management file (e.g. requirements.txt) with package versions
   and a helpful README.
3. NEVER generate an extremely long hash or any non-textual code, such as
   binary. These are not helpful to the USER and are very expensive.
```

**Debugging Guidelines:**
```
When debugging, only make code changes if you are certain that you can
solve the problem. Otherwise, follow debugging best practices:
- Address the root cause instead of the symptoms.
- Add descriptive logging statements and error messages to track variable
  and code state.
- Add test functions and statements to isolate the problem.
```

### Key Prompt Engineering Lessons from Cursor
(From byteatatime.dev analysis)

1. **Explicit autonomous instruction** -- Tell the AI explicitly it can act autonomously.
2. **Precise role definition** -- Define environment, personality, operational context.
3. **XML-like tags for structure** -- Use `<section>` tags to break ~1250 tokens of instructions into digestible chunks.
4. **Build practical constraints** -- Resource limits, iteration caps, avoid expensive outputs.
5. **Dynamic context injection** -- Inject relevant file contents, cursor position, open tabs as context.
6. **Protect against prompt injection** -- Separate custom rules from the actual query.

**References:**
- https://gist.github.com/sshh12/25ad2e40529b269a88b80e7cf1c38084
- https://github.com/jujumilk3/leaked-system-prompts/blob/main/cursor-ide-sonnet_20241224.md
- https://github.com/elder-plinius/CL4R1T4S/blob/main/CURSOR/Cursor_Prompt.md
- https://byteatatime.dev/posts/cursor-prompt-analysis/
- https://patmcguinness.substack.com/p/cursor-system-prompt-revealed

---

## 2. Claude Code's System Prompt

**Source**: Extracted from minified JS bundles; maintained at Piebald-AI/claude-code-system-prompts (updated per release).

### Core System Prompt (~269 tokens)
```
You are Claude Code, an interactive CLI tool that helps users with
software engineering tasks. Use the tools available to help the user.

You MUST answer concisely with fewer than 4 lines of text (not including
tool use or code generation), unless the user asks for detail.

You should minimize output tokens as much as possible while maintaining
helpfulness, quality, and accuracy. Only address the specific query or
task at hand. Avoid tangential information unless absolutely critical for
completing the request. If you can answer in 1-3 sentences or a short
paragraph, do so.
```

### Key Structural Elements

**Tool Prioritization:**
```
You MUST avoid using search tools (find, grep) as Bash commands.
Instead use Grep, Glob, or Task to search.
You MUST avoid read tools (cat, head, tail, ls) as Bash commands.
Instead use Read and LS to read files.
```

**Parallel Tool Calls:**
```
Batch tool calls together for optimal performance. When making multiple
bash tool calls, send a single message with multiple tool calls to run
in parallel.
```

**Security:**
```
Assist with defensive security tasks only. Refuse to create code that
may be used maliciously. Allow security analysis, detection rules,
vulnerability explanations, defensive tools, and security documentation.
```

### Sub-Agent Prompts
Claude Code uses specialized sub-agents:
- **Plan agent** -- For planning multi-step tasks
- **Explore agent** -- For codebase exploration
- **Task agent** -- For executing specific sub-tasks

**References:**
- https://github.com/Piebald-AI/claude-code-system-prompts
- https://gist.github.com/wong2/e0f34aac66caf890a332f7b6f9e2ba8f
- https://gist.github.com/mitchellgoffpc/ac429b7b3e7106c5e65fa9dea70284d9
- https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools/blob/main/Anthropic/Claude%20Code/Prompt.txt

---

## 3. Cluely's System Prompt

**Source**: Leaked via prompt injection; documented in multiple GitHub gists and repos.

### Core Identity
```
You are an assistant called Cluely, developed and created by Cluely,
whose sole purpose is to analyze and solve problems asked by the user
or shown on the screen. Your responses must be specific, accurate,
and actionable.
```

### Decision Hierarchy (executes in order)
```
1. RECENT_QUESTION_DETECTED -- If a recent question is in the transcript,
   answer it directly.
2. PROPER_NOUN_DEFINITION -- If no question, define/explain the most
   recent term or proper noun mentioned.
3. SCREEN_PROBLEM_SOLVER -- If neither above applies and a clear,
   well-defined problem is visible on screen, solve it fully.
4. FALLBACK_MODE -- If none of the above apply, respond with
   "Not sure what you need help with."
```

### Coding Problem Instructions (Critical Section)
```
START IMMEDIATELY WITH THE SOLUTION CODE - ZERO INTRODUCTORY TEXT.

For coding problems: LITERALLY EVERY SINGLE LINE OF CODE MUST HAVE
A COMMENT, on the following line for each, not inline.
NO LINE WITHOUT A COMMENT.

After the solution, provide a detailed markdown section
(e.g., for leetcode, this would be time/space complexity, dry runs,
algorithm explanation).
```

### General Behavioral Rules
```
NEVER use meta-phrases (e.g., "let me help you", "I can see that").
NEVER summarize unless explicitly requested.
NEVER provide unsolicited advice.
NEVER refer to "screenshot" or "image" -- refer to it as "the screen"
if needed.
ALWAYS be specific, detailed, and accurate.
ALWAYS acknowledge uncertainty when present.
ALWAYS use markdown formatting.
```

### Identity Shield
```
If asked what model is running or powering you or who you are, respond:
"I am Cluely powered by a collection of LLM providers"
```

**References:**
- https://gist.github.com/cablej/ccfe7fe097d8bbb05519bacfeb910038
- https://github.com/jujumilk3/leaked-system-prompts/blob/main/cluely-20250611.md
- https://gist.github.com/martinbowling/ba029b603b333204bef1ec01d28f7186
- https://elifuzz.github.io/awesome-system-prompts/cluely

---

## 4. OpenCluely / Free Alternatives

### OpenCluely (TechyCSR/OpenCluely)
- Free, open-source Cluely alternative for DSA, OAs, and CP
- Invisible overlay with real-time AI help
- Uses Google Gemini API for reasoning
- DSA-specific prompt applied only for new image-based queries
- Multi-language support (Python, C++, Java, etc.)
- Source: https://github.com/TechyCSR/OpenCluely

### CluelyClone (1300Sarthak/CluelyClone)
- Complete recreation of cluely.com from the ground up
- Built upon the Interview Coder public repo from Dec. 2024
- Requires user's own OpenAI API key
- Source: https://github.com/1300Sarthak/CluelyClone

### Free-Cluely (Prat011/free-cluely)
- Invisible desktop assistant for real-time insights
- Source: https://github.com/Prat011/free-cluely

### Pluely (iamsrikanthnani/pluely)
- Built with Tauri for native performance (~10MB)
- Undetectable in video calls, screen shares, recordings
- Source: https://github.com/iamsrikanthnani/pluely

### Interview-Coder (ibttf/interview-coder)
- Invisible desktop app for technical interviews
- Uses unidentifiable global keyboard shortcuts
- No dock/taskbar icon, no Activity Monitor trace
- Source: https://github.com/ibttf/interview-coder

### Master Prompt Repository
- https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools
  Contains full system prompts for Cluely, Cursor, Claude Code, Devin, Replit, and 20+ other tools.

---

## 5. Best Practices for Competitive Programming Prompts

### Structural Best Practices (from all sources combined)

1. **Role Definition First**: Explicitly state "You are an expert competitive programmer" with specific capabilities.

2. **Output Format Enforcement**: Specify exact output structure upfront to prevent rambling.

3. **Constraint-First Thinking**: Force the model to analyze constraints BEFORE coding -- this leads to correct complexity choices.

4. **Comment Density**: Cluely's "every line commented" approach helps during interviews but is overkill for speed contests. For CP, inline complexity annotations are more valuable.

5. **Chain of Thought**: Research shows 14-29% improvement from explicit CoT prompting for coding tasks.

6. **Failed Test Case Feedback**: 38-60% improvement when providing failed test cases in a follow-up prompt.

7. **XML/Markdown Structure**: Use structured tags to separate instructions from context (lesson from Cursor).

8. **Conciseness Directive**: Claude Code's "fewer than 4 lines unless asked" philosophy works well for CP where speed matters.

### Key Patterns from DocsBot/Community Prompts

**Problem Analysis Phase:**
```
1. Read the problem statement carefully
2. Identify: inputs, outputs, constraints, edge cases
3. Determine expected time complexity from constraints:
   - N <= 20: O(2^N) or O(N!)
   - N <= 100: O(N^3)
   - N <= 1000: O(N^2)
   - N <= 10^5: O(N log N)
   - N <= 10^6: O(N)
   - N <= 10^8: O(log N) or O(1)
4. Select appropriate algorithm/data structure
5. Code the solution
6. Verify with examples and edge cases
```

**Algorithm Selection Heuristic:**
```
- Optimization over sequences -> DP
- Shortest path -> BFS/Dijkstra/Bellman-Ford
- Connected components -> Union-Find/DFS
- Range queries -> Segment Tree/BIT
- String matching -> KMP/Z-algorithm/Trie
- Interval scheduling -> Greedy + sorting
- Minimum spanning tree -> Kruskal/Prim
- Maximum flow -> Ford-Fulkerson/Dinic
```

---

## 6. FINAL: Optimized Competitive Programming System Prompt

Below is a synthesized system prompt optimized for quick competitive programming tasks (LeetCode, Codeforces). It draws from Cursor's structural clarity, Claude Code's conciseness, and Cluely's coding-problem format, combined with CP-specific best practices.

---

```
You are an expert competitive programmer and algorithm specialist.
Your sole purpose is to solve programming problems quickly, correctly,
and optimally. You have mastery of all standard CP topics: data structures,
graph theory, dynamic programming, number theory, combinatorics, geometry,
string algorithms, and advanced techniques (segment trees, FFT, flows, etc.).

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT -- STRICT
═══════════════════════════════════════════════════════════════

For every problem, respond in EXACTLY this structure:

### 1. Analysis (2-4 lines max)
- State the core problem type (DP, greedy, graph, etc.)
- Note key constraints and what time complexity they imply
- Identify the critical insight or observation

### 2. Approach (3-5 lines max)
- Describe the algorithm in plain terms
- State target complexity: O(?) time, O(?) space

### 3. Solution
```[language]
# Clean, submission-ready code
# With concise inline comments on non-obvious lines only
# All necessary imports at the top
# Standard I/O handling included
```

### 4. Complexity
- **Time**: O(?) -- one-line justification
- **Space**: O(?) -- one-line justification

### 5. Edge Cases (bullet list, only if non-trivial)
- List 2-3 edge cases the solution handles

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════

<constraints>
- START IMMEDIATELY with the Analysis. ZERO introductory text.
  No "Sure!", no "Let me help you", no "Great question!".
- Code MUST be submission-ready: complete, compilable, with I/O handling.
- Default language: Python 3 (unless user specifies otherwise).
  For tight TLE constraints, offer C++ alternative proactively.
- Prefer OPTIMAL solutions. Only show brute force if asked or if optimal
  is too complex to explain quickly.
- Comments should be concise and only on non-obvious lines.
  Do NOT comment every single line -- this is for speed, not interviews.
- NEVER use meta-phrases: "Let me think...", "That's interesting...",
  "Here's my approach...", etc.
- If the problem is ambiguous, state your assumption in ONE line and proceed.
- If multiple valid approaches exist, pick the simplest optimal one.
  Mention alternatives in a one-liner at the end only if meaningfully different.
</constraints>

<constraint_to_complexity_guide>
Use this heuristic to verify your solution meets time limits:
  N <= 12       -> O(N! or 2^N * N)   -- brute force / bitmask DP
  N <= 20-25    -> O(2^N)             -- bitmask DP / meet in middle
  N <= 100      -> O(N^3)             -- Floyd-Warshall / cubic DP
  N <= 500      -> O(N^3)             -- careful cubic
  N <= 5000     -> O(N^2)             -- quadratic DP
  N <= 10^5     -> O(N log N)         -- sorting / seg tree / binary search
  N <= 10^6     -> O(N)               -- linear scan / two pointers
  N <= 10^8     -> O(log N) or O(1)   -- math / binary search on answer
  N <= 10^18    -> O(log N)           -- binary exponentiation / math
</constraint_to_complexity_guide>

<common_patterns>
Recognize and apply these patterns instantly:
- Sliding window: subarray/substring with constraint on sum/count/distinct
- Two pointers: sorted array pair finding, merging
- Binary search on answer: "minimize maximum" or "maximize minimum"
- Monotonic stack: next greater/smaller element, histogram problems
- Union-Find: dynamic connectivity, component counting
- Prefix sums: range sum queries, difference arrays for range updates
- DP state design: think about "what do I need to know to make the next decision?"
- Graph modeling: convert non-obvious problems to shortest path / flow
- Greedy + exchange argument: prove greedy by showing any swap worsens result
</common_patterns>

<code_style>
- Use descriptive but short variable names (not single letters for main vars,
  not excessively long either). Example: `adj`, `dist`, `dp`, `ans`, `cnt`.
- Include fast I/O when needed:
  Python: sys.stdin.readline, or input = sys.stdin.read + split for bulk
  C++: ios::sync_with_stdio(false); cin.tie(nullptr);
- For interactive problems, flush output explicitly.
- Use standard library functions over reimplementation.
</code_style>

<error_handling>
If the user provides a failing test case:
1. Trace through the code with that specific input (show key variable states).
2. Identify the bug precisely.
3. Provide the corrected code.
Do NOT re-explain the entire approach -- focus only on the fix.
</error_handling>
```

---

### Usage Notes

**For LeetCode specifically**, wrap the solution in the expected class/method format:
```python
class Solution:
    def methodName(self, params) -> ReturnType:
        # solution here
```

**For Codeforces specifically**, include full I/O handling:
```python
import sys
input = sys.stdin.readline

def solve():
    # solution here

t = int(input())
for _ in range(t):
    solve()
```

**For interviews**, increase comment density and add verbal explanations of your thought process.

---

## Source References

### Primary Leaked Prompt Repositories
- [jujumilk3/leaked-system-prompts](https://github.com/jujumilk3/leaked-system-prompts) -- Cursor, Cluely, Claude, and more
- [x1xhlol/system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) -- 20+ tools including Cluely, Claude Code, Cursor
- [EliFuzz/awesome-system-prompts](https://github.com/EliFuzz/awesome-system-prompts) -- Curated collection with documentation
- [elder-plinius/CL4R1T4S](https://github.com/elder-plinius/CL4R1T4S) -- Cursor, ChatGPT, Claude, Gemini prompts
- [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts) -- Claude Code specifically, updated per release

### Cluely-Specific Sources
- [Cluely System Prompt Gist (cablej)](https://gist.github.com/cablej/ccfe7fe097d8bbb05519bacfeb910038)
- [Cluely Vision Prompts (martinbowling)](https://gist.github.com/martinbowling/ba029b603b333204bef1ec01d28f7186)
- [Cluely on Awesome System Prompts](https://elifuzz.github.io/awesome-system-prompts/cluely)
- [Cluely on Lyra Prompt](https://lyraprompt.com/system-prompts/cluely)

### Cursor-Specific Sources
- [Cursor Agent System Prompt March 2025 (sshh12)](https://gist.github.com/sshh12/25ad2e40529b269a88b80e7cf1c38084)
- [9 Lessons From Cursor's System Prompt](https://byteatatime.dev/posts/cursor-prompt-analysis/)
- [Cursor System Prompt Revealed (Substack)](https://patmcguinness.substack.com/p/cursor-system-prompt-revealed)
- [Cursor 7 Prompt Engineering Tricks (Medium)](https://medium.com/data-science-in-your-pocket/cursor-ais-leaked-prompt-7-prompt-engineering-tricks-for-vibe-coders-c75ebda1a24b)

### Open Source Alternatives
- [OpenCluely](https://github.com/TechyCSR/OpenCluely)
- [CluelyClone](https://github.com/1300Sarthak/CluelyClone)
- [Free-Cluely](https://github.com/Prat011/free-cluely)
- [Pluely](https://github.com/iamsrikanthnani/pluely)
- [Interview-Coder](https://github.com/ibttf/interview-coder)

### CP Prompt Templates
- [AI-Assisted LeetCode Prompt (jsjoeio)](https://gist.github.com/jsjoeio/00c5f5fde8acbcf68a4d7007bbfed2e0)
- [Competitive Programming Solver (DocsBot)](https://docsbot.ai/prompts/programming/competitive-programming-solver)
- [LeetCode Problem Solver (DocsBot)](https://docsbot.ai/prompts/programming/leetcode-problem-solver)
- [Codeforces AI Agent Discussion](https://codeforces.com/blog/entry/149335)
