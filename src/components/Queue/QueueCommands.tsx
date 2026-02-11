import React, { useState, useEffect, useRef } from "react"
import { IoLogOutOutline } from "react-icons/io5"

interface QueueCommandsProps {
  onTooltipVisibilityChange: (visible: boolean, height: number) => void
  screenshots: Array<{ path: string; preview: string }>
  onChatToggle: () => void
  onSettingsToggle: () => void
}

const QueueCommands: React.FC<QueueCommandsProps> = ({
  onTooltipVisibilityChange,
  screenshots,
  onChatToggle,
  onSettingsToggle
}) => {
  const [isTooltipVisible, setIsTooltipVisible] = useState(false)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [audioResult, setAudioResult] = useState<string | null>(null)
  const chunks = useRef<Blob[]>([])

  useEffect(() => {
    let tooltipHeight = 0
    if (tooltipRef.current && isTooltipVisible) {
      tooltipHeight = tooltipRef.current.offsetHeight + 10
    }
    onTooltipVisibilityChange(isTooltipVisible, tooltipHeight)
  }, [isTooltipVisible])

  const handleRecordClick = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream)
        recorder.ondataavailable = (e) => chunks.current.push(e.data)
        recorder.onstop = async () => {
          const blob = new Blob(chunks.current, { type: chunks.current[0]?.type || 'audio/webm' })
          chunks.current = []
          const reader = new FileReader()
          reader.onloadend = async () => {
            const base64Data = (reader.result as string).split(',')[1]
            try {
              const result = await window.electronAPI.analyzeAudioFromBase64(base64Data, blob.type)
              setAudioResult(result.text)
            } catch (err) {
              setAudioResult('Audio analysis failed.')
            }
          }
          reader.readAsDataURL(blob)
        }
        setMediaRecorder(recorder)
        recorder.start()
        setIsRecording(true)
      } catch (err) {
        setAudioResult('Could not start recording.')
      }
    } else {
      mediaRecorder?.stop()
      setIsRecording(false)
      setMediaRecorder(null)
    }
  }

  return (
    <div className="w-fit">
      <div className="command-bar draggable-area">
        {/* Toggle */}
        <div className="command-group">
          <span className="command-label">Show/Hide</span>
          <div className="key-combo">
            <kbd>⌘</kbd><kbd>B</kbd>
          </div>
        </div>

        <div className="command-divider" />

        {/* Screenshot */}
        <div className="command-group">
          <span className="command-label">Screenshot</span>
          <div className="key-combo">
            <kbd>⌘</kbd><kbd>H</kbd>
          </div>
        </div>

        {/* Solve */}
        {screenshots.length > 0 && (
          <>
            <div className="command-divider" />
            <div className="command-group">
              <span className="command-label">Solve</span>
              <div className="key-combo">
                <kbd>⌘</kbd><kbd>↵</kbd>
              </div>
            </div>
          </>
        )}

        <div className="command-divider" />

        {/* Voice */}
        <button
          className={`command-btn ${isRecording ? 'recording' : ''}`}
          onClick={handleRecordClick}
          type="button"
        >
          {isRecording ? (
            <span className="flex items-center gap-1">
              <span className="rec-dot" />Stop
            </span>
          ) : (
            <span>Mic</span>
          )}
        </button>

        {/* Chat */}
        <button className="command-btn" onClick={onChatToggle} type="button">
          Chat
        </button>

        {/* Models */}
        <button className="command-btn" onClick={onSettingsToggle} type="button">
          Models
        </button>

        <div className="command-divider" />

        {/* Help */}
        <div
          className="relative inline-block"
          onMouseEnter={() => setIsTooltipVisible(true)}
          onMouseLeave={() => setIsTooltipVisible(false)}
        >
          <button className="command-btn help-btn" type="button">?</button>
          {isTooltipVisible && (
            <div ref={tooltipRef} className="tooltip-panel">
              <div className="tooltip-content">
                <h4 className="tooltip-title">Keyboard Shortcuts</h4>
                <div className="tooltip-row">
                  <span>Toggle Window</span>
                  <div className="key-combo"><kbd>⌘</kbd><kbd>B</kbd></div>
                </div>
                <div className="tooltip-row">
                  <span>Take Screenshot</span>
                  <div className="key-combo"><kbd>⌘</kbd><kbd>H</kbd></div>
                </div>
                <div className="tooltip-row">
                  <span>Solve Problem</span>
                  <div className="key-combo"><kbd>⌘</kbd><kbd>↵</kbd></div>
                </div>
                <div className="tooltip-row">
                  <span>Reset</span>
                  <div className="key-combo"><kbd>⌘</kbd><kbd>R</kbd></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Quit */}
        <button
          className="command-btn quit-btn"
          title="Quit"
          onClick={() => window.electronAPI.quitApp()}
          type="button"
        >
          <IoLogOutOutline className="w-3.5 h-3.5" />
        </button>
      </div>

      {audioResult && (
        <div className="audio-result">
          <span className="font-semibold">Audio:</span> {audioResult}
        </div>
      )}
    </div>
  )
}

export default QueueCommands
