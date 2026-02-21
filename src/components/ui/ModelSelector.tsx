import React, { useState, useEffect } from 'react';

interface ModelConfig {
  provider: "ollama" | "gemini";
  model: string;
  isOllama: boolean;
}

interface ModelSelectorProps {
  onModelChange?: (provider: "ollama" | "gemini", model: string) => void;
  onChatOpen?: () => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onModelChange, onChatOpen }) => {
  const [currentConfig, setCurrentConfig] = useState<ModelConfig | null>(null);
  const [availableOllamaModels, setAvailableOllamaModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'testing' | 'success' | 'error' | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [selectedProvider, setSelectedProvider] = useState<"ollama" | "gemini">("gemini");
  const [selectedOllamaModel, setSelectedOllamaModel] = useState<string>("");
  const [ollamaUrl, setOllamaUrl] = useState<string>("http://localhost:11434");
  const [availableGeminiModels, setAvailableGeminiModels] = useState<string[]>([]);
  const [selectedGeminiModel, setSelectedGeminiModel] = useState<string>("gemini-3.1-flash-preview");

  useEffect(() => {
    loadCurrentConfig();
  }, []);

  const loadCurrentConfig = async () => {
    try {
      setIsLoading(true);
      const config = await window.electronAPI.getCurrentLlmConfig();
      setCurrentConfig(config);
      setSelectedProvider(config.provider);

      if (config.isOllama) {
        setSelectedOllamaModel(config.model);
        await loadOllamaModels();
      } else {
        setSelectedGeminiModel(config.model);
        await loadGeminiModels();
      }
    } catch (error) {
      console.error('Error loading current config:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadGeminiModels = async () => {
    try {
      const models = await window.electronAPI.getAvailableGeminiModels();
      setAvailableGeminiModels(models);
    } catch (error) {
      console.error('Error loading Gemini models:', error);
      setAvailableGeminiModels(["gemini-3.1-flash-preview", "gemini-3-flash-preview", "gemini-3-pro-preview"]);
    }
  };

  const loadOllamaModels = async () => {
    try {
      const models = await window.electronAPI.getAvailableOllamaModels();
      setAvailableOllamaModels(models);
      if (models.length > 0 && !selectedOllamaModel) {
        setSelectedOllamaModel(models[0]);
      }
    } catch (error) {
      console.error('Error loading Ollama models:', error);
      setAvailableOllamaModels([]);
    }
  };

  const testConnection = async () => {
    try {
      setConnectionStatus('testing');
      const result = await window.electronAPI.testLlmConnection();
      setConnectionStatus(result.success ? 'success' : 'error');
      if (!result.success) setErrorMessage(result.error || 'Unknown error');
    } catch (error) {
      setConnectionStatus('error');
      setErrorMessage(String(error));
    }
  };

  const handleProviderSwitch = async () => {
    try {
      setConnectionStatus('testing');
      let result;

      if (selectedProvider === 'ollama') {
        result = await window.electronAPI.switchToOllama(selectedOllamaModel, ollamaUrl);
      } else {
        result = await window.electronAPI.switchToGemini(geminiApiKey || undefined, selectedGeminiModel);
      }

      if (result.success) {
        await loadCurrentConfig();
        setConnectionStatus('success');
        onModelChange?.(selectedProvider, selectedProvider === 'ollama' ? selectedOllamaModel : selectedGeminiModel);
        setTimeout(() => onChatOpen?.(), 500);
      } else {
        setConnectionStatus('error');
        setErrorMessage(result.error || 'Switch failed');
      }
    } catch (error) {
      setConnectionStatus('error');
      setErrorMessage(String(error));
    }
  };

  const statusIndicator = () => {
    if (connectionStatus === 'testing') return <span className="status-dot testing" />;
    if (connectionStatus === 'success') return <span className="status-dot success" />;
    if (connectionStatus === 'error') return <span className="status-dot error" />;
    return null;
  };

  if (isLoading) {
    return (
      <div className="settings-panel">
        <div className="text-xs text-white/50 text-center py-4">Loading...</div>
      </div>
    );
  }

  return (
    <div className="settings-panel">
      {/* Header */}
      <div className="settings-header">
        <span className="settings-title">Model Configuration</span>
        <div className="flex items-center gap-2">
          {statusIndicator()}
          {currentConfig && (
            <span className="current-model-badge">
              {currentConfig.model}
            </span>
          )}
        </div>
      </div>

      {/* Provider Toggle */}
      <div className="provider-toggle">
        <button
          onClick={() => { setSelectedProvider('gemini'); loadGeminiModels(); }}
          className={`provider-btn ${selectedProvider === 'gemini' ? 'active' : ''}`}
        >
          Gemini
        </button>
        <button
          onClick={() => { setSelectedProvider('ollama'); loadOllamaModels(); }}
          className={`provider-btn ${selectedProvider === 'ollama' ? 'active' : ''}`}
        >
          Ollama
        </button>
      </div>

      {/* Provider Settings */}
      {selectedProvider === 'gemini' ? (
        <div className="settings-fields">
          <div className="field-group">
            <label className="field-label">Model</label>
            <select
              value={selectedGeminiModel}
              onChange={(e) => setSelectedGeminiModel(e.target.value)}
              className="field-select"
            >
              {(availableGeminiModels.length > 0
                ? availableGeminiModels
                : ["gemini-3.1-flash-preview", "gemini-3-flash-preview", "gemini-3-pro-preview"]
              ).map((model) => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
          <div className="field-group">
            <label className="field-label">API Key (optional if set)</label>
            <input
              type="password"
              placeholder="Enter to update..."
              value={geminiApiKey}
              onChange={(e) => setGeminiApiKey(e.target.value)}
              className="field-input"
            />
          </div>
        </div>
      ) : (
        <div className="settings-fields">
          <div className="field-group">
            <label className="field-label">Ollama URL</label>
            <input
              type="url"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              className="field-input"
            />
          </div>
          <div className="field-group">
            <div className="flex items-center justify-between">
              <label className="field-label">Model</label>
              <button onClick={loadOllamaModels} className="refresh-btn" title="Refresh">
                Refresh
              </button>
            </div>
            {availableOllamaModels.length > 0 ? (
              <select
                value={selectedOllamaModel}
                onChange={(e) => setSelectedOllamaModel(e.target.value)}
                className="field-select"
              >
                {availableOllamaModels.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            ) : (
              <div className="text-[10px] text-white/40 py-1">
                No models found. Ensure Ollama is running.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error message */}
      {connectionStatus === 'error' && (
        <div className="error-msg">{errorMessage}</div>
      )}

      {/* Actions */}
      <div className="settings-actions">
        <button
          onClick={handleProviderSwitch}
          disabled={connectionStatus === 'testing'}
          className="apply-btn"
        >
          {connectionStatus === 'testing' ? 'Applying...' : 'Apply'}
        </button>
        <button
          onClick={testConnection}
          disabled={connectionStatus === 'testing'}
          className="test-btn"
        >
          Test
        </button>
      </div>
    </div>
  );
};

export default ModelSelector;
