// ═══════════════════════════════════════════════════════════════
// OPENAI WHISPER PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{SttConfig, SttProvider, SttProviderType, SttResult, SttWord, audio_to_wav, validate_audio_data};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OpenAI Whisper STT provider
pub struct OpenAiSttProvider {
    client: Client,
    api_base: String,
}

impl OpenAiSttProvider {
    /// Create a new OpenAI Whisper provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://api.openai.com/v1".to_string(),
        }
    }

    /// Create a new provider with custom API base
    pub fn with_api_base(api_base: String) -> Self {
        Self {
            client: Client::new(),
            api_base,
        }
    }

    /// Get the API key from environment or config
    fn get_api_key(&self, config: &SttConfig) -> Result<String> {
        config
            .api_key
            .as_ref()
            .cloned()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .context("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the model to use
    fn get_model(&self, config: &SttConfig) -> String {
        config
            .model
            .clone()
            .unwrap_or_else(|| "whisper-1".to_string())
    }

    /// Call the OpenAI Whisper API
    async fn call_whisper_api(
        &self,
        wav_data: Vec<u8>,
        api_key: &str,
        model: &str,
        config: &SttConfig,
    ) -> Result<WhisperResponse> {
        let url = format!("{}/audio/transcriptions", self.api_base);

        // Create multipart form
        let part = reqwest::multipart::Part::bytes(wav_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .unwrap();

        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", model.to_string())
            .text("language", config.language.to_lowercase())
            .text("response_format", "verbose_json");

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .multipart(form)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API request failed: {status} - {error_text}");
        }

        response
            .json::<WhisperResponse>()
            .await
            .context("Failed to parse OpenAI API response")
    }

    /// Convert Whisper response to SttResult
    fn convert_response(response: WhisperResponse, _config: &SttConfig) -> SttResult {
        let words: Vec<SttWord> = response
            .words
            .into_iter()
            .map(|w| SttWord {
                word: w.word,
                start_time: w.start,
                end_time: w.end,
                confidence: w.probability.unwrap_or(1.0),
            })
            .collect();

        SttResult {
            text: response.text,
            confidence: 0.95, // Whisper doesn't provide overall confidence
            language: Some("en".to_string()), // Could be parsed from response
            alternatives: Vec::new(), // Whisper doesn't provide alternatives
            words,
        }
    }
}

impl Default for OpenAiSttProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for OpenAiSttProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::OpenAi
    }

    async fn transcribe(&self, audio_data: &[f32], config: &SttConfig) -> Result<SttResult> {
        // Validate audio data
        validate_audio_data(audio_data)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get model
        let model = self.get_model(config);

        // Convert to WAV
        let wav_data = audio_to_wav(audio_data, config.sample_rate);

        // Call API
        let response = self
            .call_whisper_api(wav_data, &api_key, &model, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn transcribe_file(&self, file_path: &std::path::Path, config: &SttConfig) -> Result<SttResult> {
        // Read file
        let audio_data = std::fs::read(file_path)
            .context("Failed to read audio file")?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get model
        let model = self.get_model(config);

        // Call API with file data
        let part = reqwest::multipart::Part::bytes(audio_data)
            .file_name(file_path.file_name().unwrap().to_string_lossy().to_string())
            .mime_str("audio/wav")
            .unwrap();

        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("model", model.to_string())
            .text("language", config.language.to_lowercase())
            .text("response_format", "verbose_json");

        let url = format!("{}/audio/transcriptions", self.api_base);
        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API request failed: {status} - {error_text}");
        }

        let whisper_response = response
            .json::<WhisperResponse>()
            .await
            .context("Failed to parse OpenAI API response")?;

        Ok(Self::convert_response(whisper_response, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("OPENAI_API_KEY").is_ok()
    }
}

// ═══════════════════════════════════════════════════════════════
// OPENAI API TYPES
// ═══════════════════════════════════════════════════════════════

/// Whisper API response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhisperResponse {
    pub text: String,
    #[serde(default)]
    pub words: Vec<WhisperWord>,
    pub language: Option<String>,
    pub duration: Option<f32>,
}

/// Word-level timestamp from Whisper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhisperWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub probability: Option<f32>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_new() {
        let provider = OpenAiSttProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_default() {
        let provider = OpenAiSttProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_with_api_base() {
        let provider = OpenAiSttProvider::with_api_base("https://api.example.com".to_string());
        assert_eq!(provider.provider_type(), SttProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_name() {
        let provider = OpenAiSttProvider::new();
        assert_eq!(provider.name(), "OpenAI Whisper");
    }

    #[test]
    fn test_whisper_response_empty() {
        let response = WhisperResponse {
            text: String::new(),
            words: Vec::new(),
            language: None,
            duration: None,
        };

        let result = OpenAiSttProvider::convert_response(response, &SttConfig::default());
        assert!(result.text.is_empty());
        assert!(result.words.is_empty());
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_whisper_response_with_words() {
        let response = WhisperResponse {
            text: "hello world".to_string(),
            words: vec![
                WhisperWord {
                    word: "hello".to_string(),
                    start: 0.0,
                    end: 0.5,
                    probability: Some(0.98),
                },
                WhisperWord {
                    word: "world".to_string(),
                    start: 0.5,
                    end: 1.0,
                    probability: Some(0.95),
                },
            ],
            language: Some("en".to_string()),
            duration: Some(1.0),
        };

        let result = OpenAiSttProvider::convert_response(response, &SttConfig::default());
        assert_eq!(result.text, "hello world");
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].word, "hello");
        assert_eq!(result.words[1].word, "world");
    }

    #[tokio::test]
    async fn test_openai_provider_no_api_key() {
        // Temporarily remove API key
        let _guard = std::env::var("OPENAI_API_KEY").ok();

        std::env::remove_var("OPENAI_API_KEY");

        let provider = OpenAiSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.1; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));

        // Restore original if it existed
        if let Some(key) = _guard {
            std::env::set_var("OPENAI_API_KEY", key);
        }
    }

    #[tokio::test]
    async fn test_openai_provider_empty_audio() {
        let provider = OpenAiSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_openai_provider_invalid_audio() {
        let provider = OpenAiSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.5, f32::NAN, 0.3];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_openai_provider_is_available() {
        let provider = OpenAiSttProvider::new();
        // This test doesn't require an actual API key
        // It just checks the method doesn't panic
        let _ = provider.is_available().await;
    }

    #[test]
    fn test_get_model_default() {
        let provider = OpenAiSttProvider::new();
        let config = SttConfig {
            model: None,
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "whisper-1");
    }

    #[test]
    fn test_get_model_custom() {
        let provider = OpenAiSttProvider::new();
        let config = SttConfig {
            model: Some("whisper-large-v3".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "whisper-large-v3");
    }

    #[test]
    fn test_whisper_word_serialization() {
        let word = WhisperWord {
            word: "hello".to_string(),
            start: 0.0,
            end: 0.5,
            probability: Some(0.98),
        };

        let json = serde_json::to_string(&word).unwrap();
        let parsed: WhisperWord = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.word, "hello");
        assert_eq!(parsed.start, 0.0);
        assert_eq!(parsed.end, 0.5);
        assert_eq!(parsed.probability, Some(0.98));
    }

    #[test]
    fn test_whisper_word_no_probability() {
        let word = WhisperWord {
            word: "hello".to_string(),
            start: 0.0,
            end: 0.5,
            probability: None,
        };

        let json = serde_json::to_string(&word).unwrap();
        let parsed: WhisperWord = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.word, "hello");
        assert_eq!(parsed.probability, None);
    }

    #[tokio::test]
    async fn test_openai_provider_too_quiet_audio() {
        let provider = OpenAiSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.0001; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too quiet"));
    }
}
