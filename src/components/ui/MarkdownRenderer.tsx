import React, { useMemo } from "react"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { dracula } from "react-syntax-highlighter/dist/esm/styles/prism"

interface MarkdownRendererProps {
  content: string
}

interface Block {
  type: "code" | "text"
  content: string
  language?: string
}

const parseBlocks = (text: string): Block[] => {
  const blocks: Block[] = []
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g
  let lastIndex = 0
  let match

  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ type: "text", content: text.slice(lastIndex, match.index) })
    }
    blocks.push({
      type: "code",
      language: match[1] || "text",
      content: match[2].replace(/\n$/, ""),
    })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    blocks.push({ type: "text", content: text.slice(lastIndex) })
  }

  return blocks
}

const renderInline = (text: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = []
  // Match: bold **text**, italic *text*, inline code `code`, links [text](url)
  const inlineRegex = /(\*\*(.+?)\*\*)|(\*(.+?)\*)|(`([^`]+?)`)|(\[([^\]]+?)\]\(([^)]+?)\))/g
  let lastIdx = 0
  let m
  let key = 0

  while ((m = inlineRegex.exec(text)) !== null) {
    if (m.index > lastIdx) {
      parts.push(text.slice(lastIdx, m.index))
    }

    if (m[1]) {
      // Bold
      parts.push(<strong key={key++}>{m[2]}</strong>)
    } else if (m[3]) {
      // Italic
      parts.push(<em key={key++}>{m[4]}</em>)
    } else if (m[5]) {
      // Inline code
      parts.push(<code key={key++} className="md-inline-code">{m[6]}</code>)
    } else if (m[7]) {
      // Link
      parts.push(
        <a key={key++} href={m[9]} target="_blank" rel="noopener noreferrer" className="md-link">
          {m[8]}
        </a>
      )
    }
    lastIdx = m.index + m[0].length
  }

  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx))
  }

  return parts
}

const renderTextBlock = (text: string): React.ReactNode[] => {
  const lines = text.split("\n")
  const elements: React.ReactNode[] = []
  let listItems: { ordered: boolean; text: string }[] = []
  let key = 0

  const flushList = () => {
    if (listItems.length === 0) return
    const ordered = listItems[0].ordered
    const Tag = ordered ? "ol" : "ul"
    elements.push(
      <Tag key={key++} className={ordered ? "md-ol" : "md-ul"}>
        {listItems.map((item, i) => (
          <li key={i}>{renderInline(item.text)}</li>
        ))}
      </Tag>
    )
    listItems = []
  }

  for (const line of lines) {
    const trimmed = line.trim()

    // Empty line
    if (!trimmed) {
      flushList()
      continue
    }

    // Headers
    const headerMatch = trimmed.match(/^(#{1,4})\s+(.+)$/)
    if (headerMatch) {
      flushList()
      const level = headerMatch[1].length
      const Tag = `h${level}` as keyof JSX.IntrinsicElements
      elements.push(
        <Tag key={key++} className={`md-h${level}`}>
          {renderInline(headerMatch[2])}
        </Tag>
      )
      continue
    }

    // Unordered list
    const ulMatch = trimmed.match(/^[-*+]\s+(.+)$/)
    if (ulMatch) {
      listItems.push({ ordered: false, text: ulMatch[1] })
      continue
    }

    // Ordered list
    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/)
    if (olMatch) {
      listItems.push({ ordered: true, text: olMatch[1] })
      continue
    }

    // Regular paragraph text
    flushList()
    elements.push(
      <p key={key++} className="md-p">
        {renderInline(trimmed)}
      </p>
    )
  }

  flushList()
  return elements
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  const rendered = useMemo(() => {
    const blocks = parseBlocks(content)

    return blocks.map((block, i) => {
      if (block.type === "code") {
        return (
          <div key={i} className="md-code-block">
            <div className="md-code-header">
              <span className="md-code-lang">{block.language}</span>
              <button
                className="md-code-copy"
                onClick={() => navigator.clipboard.writeText(block.content)}
                title="Copy code"
              >
                Copy
              </button>
            </div>
            <SyntaxHighlighter
              language={block.language}
              style={dracula}
              customStyle={{
                margin: 0,
                borderRadius: "0 0 6px 6px",
                fontSize: "11px",
                padding: "10px 12px",
                background: "rgba(0,0,0,0.4)",
              }}
              wrapLongLines
            >
              {block.content}
            </SyntaxHighlighter>
          </div>
        )
      }

      return <div key={i}>{renderTextBlock(block.content)}</div>
    })
  }, [content])

  return <div className="md-renderer">{rendered}</div>
}

export default MarkdownRenderer
