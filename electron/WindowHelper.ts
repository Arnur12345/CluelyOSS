
import { BrowserWindow, screen } from "electron"
import { AppState } from "main"
import path from "node:path"
import { exec } from "node:child_process"

const isDev = process.env.NODE_ENV === "development"

const startUrl = isDev
  ? "http://localhost:5180"
  : `file://${path.join(__dirname, "../dist/index.html")}`

// Windows display affinity constants
const WDA_NONE = 0x00000000
const WDA_MONITOR = 0x00000001
const WDA_EXCLUDEFROMCAPTURE = 0x00000011

export class WindowHelper {
  private mainWindow: BrowserWindow | null = null
  private isWindowVisible: boolean = false
  private windowPosition: { x: number; y: number } | null = null
  private windowSize: { width: number; height: number } | null = null
  private appState: AppState

  // Initialize with explicit number type and 0 value
  private screenWidth: number = 0
  private screenHeight: number = 0
  private step: number = 0
  private currentX: number = 0
  private currentY: number = 0

  // Content protection
  private contentProtectionInterval: ReturnType<typeof setInterval> | null = null
  private nativeAffinityApplied: boolean = false

  constructor(appState: AppState) {
    this.appState = appState
  }

  /**
   * Applies content protection using multiple strategies:
   * 1. Electron's setContentProtection(true) - uses WDA_EXCLUDEFROMCAPTURE on Win10 2004+
   * 2. Direct Win32 API call via PowerShell as fallback for when Electron's method fails
   *    (known to fail after hide/show cycles: https://github.com/electron/electron/issues/29085)
   */
  private ensureContentProtection(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) return

    // Re-apply Electron's built-in content protection
    this.mainWindow.setContentProtection(true)

    // On Windows, also call the native Win32 API directly as a fallback
    // This ensures WDA_EXCLUDEFROMCAPTURE is set even when Electron's method silently fails
    if (process.platform === "win32") {
      this.applyNativeDisplayAffinity()
    }
  }

  /**
   * Calls SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) directly via PowerShell.
   * This bypasses Electron's internal state management which can lose the flag after hide/show.
   * WDA_EXCLUDEFROMCAPTURE (0x11) completely removes the window from all screen capture on Win10 2004+.
   * See: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowdisplayaffinity
   */
  private applyNativeDisplayAffinity(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) return
    if (process.platform !== "win32") return

    try {
      const hwndBuffer = this.mainWindow.getNativeWindowHandle()
      // HWND is a pointer-sized value; on Windows (even x64) window handles fit in 32 bits
      const hwnd = hwndBuffer.readUInt32LE(0)

      // PowerShell script to call SetWindowDisplayAffinity via P/Invoke
      // Using EncodedCommand to avoid quoting/escaping issues
      const psScript = [
        `Add-Type -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetWindowDisplayAffinity(IntPtr hWnd, uint dwAffinity);' -Name User32 -Namespace Win32`,
        `[Win32.User32]::SetWindowDisplayAffinity([IntPtr]${hwnd}, ${WDA_EXCLUDEFROMCAPTURE})`
      ].join("; ")

      const encodedCommand = Buffer.from(psScript, "utf16le").toString("base64")

      exec(
        `powershell -NoProfile -NonInteractive -EncodedCommand ${encodedCommand}`,
        { windowsHide: true, timeout: 10000 },
        (error, stdout) => {
          if (error) {
            console.error("Native SetWindowDisplayAffinity failed:", error.message)
          } else {
            const result = stdout.trim()
            if (result === "True") {
              this.nativeAffinityApplied = true
              console.log("Native SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) applied successfully")
            } else {
              console.warn("Native SetWindowDisplayAffinity returned:", result)
            }
          }
        }
      )
    } catch (error) {
      console.error("Error calling native display affinity:", error)
    }
  }

  /**
   * Starts a periodic interval that re-applies content protection.
   * This guards against protection being silently lost due to:
   * - Electron bugs after hide/show cycles
   * - OS-level resets of window display affinity
   */
  private startContentProtectionInterval(): void {
    if (this.contentProtectionInterval) return

    this.contentProtectionInterval = setInterval(() => {
      if (this.mainWindow && !this.mainWindow.isDestroyed() && this.isWindowVisible) {
        this.mainWindow.setContentProtection(true)
      }
    }, 3000)
  }

  private stopContentProtectionInterval(): void {
    if (this.contentProtectionInterval) {
      clearInterval(this.contentProtectionInterval)
      this.contentProtectionInterval = null
    }
  }

  public setWindowDimensions(width: number, height: number): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) return

    // Get current window position
    const [currentX, currentY] = this.mainWindow.getPosition()

    // Get screen dimensions
    const primaryDisplay = screen.getPrimaryDisplay()
    const workArea = primaryDisplay.workAreaSize

    // Use 75% width if debugging has occurred, otherwise use 60%
    const maxAllowedWidth = Math.floor(
      workArea.width * (this.appState.getHasDebugged() ? 0.75 : 0.5)
    )

    // Ensure width doesn't exceed max allowed width and height is reasonable
    const newWidth = Math.min(width + 32, maxAllowedWidth)
    const newHeight = Math.ceil(height)

    // Center the window horizontally if it would go off screen
    const maxX = workArea.width - newWidth
    const newX = Math.min(Math.max(currentX, 0), maxX)

    // Update window bounds
    this.mainWindow.setBounds({
      x: newX,
      y: currentY,
      width: newWidth,
      height: newHeight
    })

    // Update internal state
    this.windowPosition = { x: newX, y: currentY }
    this.windowSize = { width: newWidth, height: newHeight }
    this.currentX = newX
  }

  public createWindow(): void {
    if (this.mainWindow !== null) return

    const primaryDisplay = screen.getPrimaryDisplay()
    const workArea = primaryDisplay.workAreaSize
    this.screenWidth = workArea.width
    this.screenHeight = workArea.height

    const windowSettings: Electron.BrowserWindowConstructorOptions = {
      width: 400,
      height: 600,
      minWidth: 300,
      minHeight: 200,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: true,
        preload: path.join(__dirname, "preload.js")
      },
      show: false, // Start hidden, then show after setup
      alwaysOnTop: true,
      frame: false,
      transparent: true,
      fullscreenable: false,
      hasShadow: false,
      backgroundColor: "#00000000",
      focusable: true,
      resizable: true,
      movable: true,
      x: 100, // Start at a visible position
      y: 100
    }

    this.mainWindow = new BrowserWindow(windowSettings)
    // this.mainWindow.webContents.openDevTools()

    // Apply content protection immediately after window creation
    this.ensureContentProtection()

    if (process.platform === "darwin") {
      this.mainWindow.setVisibleOnAllWorkspaces(true, {
        visibleOnFullScreen: true
      })
      this.mainWindow.setHiddenInMissionControl(true)
      this.mainWindow.setAlwaysOnTop(true, "floating")
    }
    if (process.platform === "linux") {
      // Linux-specific optimizations for better compatibility
      if (this.mainWindow.setHasShadow) {
        this.mainWindow.setHasShadow(false)
      }
      // Keep window focusable on Linux for proper interaction
      this.mainWindow.setFocusable(true)
    }
    this.mainWindow.setSkipTaskbar(true)
    this.mainWindow.setAlwaysOnTop(true)

    this.mainWindow.loadURL(startUrl).catch((err) => {
      console.error("Failed to load URL:", err)
    })

    // Show window after loading URL and center it
    this.mainWindow.once('ready-to-show', () => {
      if (this.mainWindow) {
        // Center the window first
        this.centerWindow()
        this.mainWindow.show()
        this.mainWindow.focus()
        this.mainWindow.setAlwaysOnTop(true)

        // Re-apply content protection after first show
        this.ensureContentProtection()

        // Start periodic re-application of content protection
        this.startContentProtectionInterval()

        console.log("Window is now visible and centered")
      }
    })

    const bounds = this.mainWindow.getBounds()
    this.windowPosition = { x: bounds.x, y: bounds.y }
    this.windowSize = { width: bounds.width, height: bounds.height }
    this.currentX = bounds.x
    this.currentY = bounds.y

    this.setupWindowListeners()
    this.isWindowVisible = true
  }

  private setupWindowListeners(): void {
    if (!this.mainWindow) return

    this.mainWindow.on("move", () => {
      if (this.mainWindow) {
        const bounds = this.mainWindow.getBounds()
        this.windowPosition = { x: bounds.x, y: bounds.y }
        this.currentX = bounds.x
        this.currentY = bounds.y
      }
    })

    this.mainWindow.on("resize", () => {
      if (this.mainWindow) {
        const bounds = this.mainWindow.getBounds()
        this.windowSize = { width: bounds.width, height: bounds.height }
      }
    })

    // Re-apply content protection whenever the window is shown
    // This catches ALL show events, including from external code
    this.mainWindow.on("show", () => {
      this.ensureContentProtection()
    })

    this.mainWindow.on("closed", () => {
      this.stopContentProtectionInterval()
      this.mainWindow = null
      this.isWindowVisible = false
      this.windowPosition = null
      this.windowSize = null
    })
  }

  public getMainWindow(): BrowserWindow | null {
    return this.mainWindow
  }

  public isVisible(): boolean {
    return this.isWindowVisible
  }

  public hideMainWindow(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) {
      console.warn("Main window does not exist or is destroyed.")
      return
    }

    const bounds = this.mainWindow.getBounds()
    this.windowPosition = { x: bounds.x, y: bounds.y }
    this.windowSize = { width: bounds.width, height: bounds.height }
    this.mainWindow.hide()
    this.isWindowVisible = false
  }

  public showMainWindow(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) {
      console.warn("Main window does not exist or is destroyed.")
      return
    }

    if (this.windowPosition && this.windowSize) {
      this.mainWindow.setBounds({
        x: this.windowPosition.x,
        y: this.windowPosition.y,
        width: this.windowSize.width,
        height: this.windowSize.height
      })
    }

    this.mainWindow.showInactive()

    // Critical: re-apply content protection after show
    // hide() → showInactive() is known to reset WDA_EXCLUDEFROMCAPTURE
    this.ensureContentProtection()

    this.isWindowVisible = true
  }

  public toggleMainWindow(): void {
    if (this.isWindowVisible) {
      this.hideMainWindow()
    } else {
      this.showMainWindow()
    }
  }

  private centerWindow(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) {
      return
    }

    const primaryDisplay = screen.getPrimaryDisplay()
    const workArea = primaryDisplay.workAreaSize

    // Get current window size or use defaults
    const windowBounds = this.mainWindow.getBounds()
    const windowWidth = windowBounds.width || 400
    const windowHeight = windowBounds.height || 600

    // Calculate center position
    const centerX = Math.floor((workArea.width - windowWidth) / 2)
    const centerY = Math.floor((workArea.height - windowHeight) / 2)

    // Set window position
    this.mainWindow.setBounds({
      x: centerX,
      y: centerY,
      width: windowWidth,
      height: windowHeight
    })

    // Update internal state
    this.windowPosition = { x: centerX, y: centerY }
    this.windowSize = { width: windowWidth, height: windowHeight }
    this.currentX = centerX
    this.currentY = centerY
  }

  public centerAndShowWindow(): void {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) {
      console.warn("Main window does not exist or is destroyed.")
      return
    }

    this.centerWindow()
    this.mainWindow.show()
    this.mainWindow.focus()
    this.mainWindow.setAlwaysOnTop(true)

    // Re-apply content protection after show
    this.ensureContentProtection()

    this.isWindowVisible = true

    console.log(`Window centered and shown`)
  }

  // New methods for window movement
  public moveWindowRight(): void {
    if (!this.mainWindow) return

    const windowWidth = this.windowSize?.width || 0
    const halfWidth = windowWidth / 2

    // Ensure currentX and currentY are numbers
    this.currentX = Number(this.currentX) || 0
    this.currentY = Number(this.currentY) || 0

    this.currentX = Math.min(
      this.screenWidth - halfWidth,
      this.currentX + this.step
    )
    this.mainWindow.setPosition(
      Math.round(this.currentX),
      Math.round(this.currentY)
    )
  }

  public moveWindowLeft(): void {
    if (!this.mainWindow) return

    const windowWidth = this.windowSize?.width || 0
    const halfWidth = windowWidth / 2

    // Ensure currentX and currentY are numbers
    this.currentX = Number(this.currentX) || 0
    this.currentY = Number(this.currentY) || 0

    this.currentX = Math.max(-halfWidth, this.currentX - this.step)
    this.mainWindow.setPosition(
      Math.round(this.currentX),
      Math.round(this.currentY)
    )
  }

  public moveWindowDown(): void {
    if (!this.mainWindow) return

    const windowHeight = this.windowSize?.height || 0
    const halfHeight = windowHeight / 2

    // Ensure currentX and currentY are numbers
    this.currentX = Number(this.currentX) || 0
    this.currentY = Number(this.currentY) || 0

    this.currentY = Math.min(
      this.screenHeight - halfHeight,
      this.currentY + this.step
    )
    this.mainWindow.setPosition(
      Math.round(this.currentX),
      Math.round(this.currentY)
    )
  }

  public moveWindowUp(): void {
    if (!this.mainWindow) return

    const windowHeight = this.windowSize?.height || 0
    const halfHeight = windowHeight / 2

    // Ensure currentX and currentY are numbers
    this.currentX = Number(this.currentX) || 0
    this.currentY = Number(this.currentY) || 0

    this.currentY = Math.max(-halfHeight, this.currentY - this.step)
    this.mainWindow.setPosition(
      Math.round(this.currentX),
      Math.round(this.currentY)
    )
  }
}
