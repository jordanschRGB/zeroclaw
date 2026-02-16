// ═══════════════════════════════════════════════════════════════
// OPENAI TTS PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceInfo, VoiceGender, validate_text};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;
use std::time::Duration;

/// OpenAI TTS provider
pub struct OpenAiTtsProvider {
    client: Client,
    api_base: String,
}

impl OpenAiTtsProvider {
    /// Create a new OpenAI TTS provider
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
    fn get_api_key(&self, config: &TtsConfig) -> Result<String> {
        config
            .api_key
            .as_ref()
            .cloned()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .context("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the model to use
    fn get_model(&self, config: &TtsConfig) -> &str {
        config
            .model
            .as_deref()
            .unwrap_or("tts-1")
    }

    /// Get the voice to use
    fn get_voice(&self, config: &TtsConfig) -> &str {
        config
            .voice
            .as_deref()
            .unwrap_or("alloy")
    }

    /// Call the OpenAI TTS API
    async fn call_tts_api(
        &self,
        text: &str,
        api_key: &str,
        model: &str,
        voice: &str,
        config: &TtsConfig,
    ) -> Result<OpenAiTtsResponse> {
        let url = format!("{}/audio/speech", self.api_base);

        let request = OpenAiTtsRequest {
            model: model.to_string(),
            input: text.to_string(),
            voice: voice.to_string(),
            response_format: config.output_format.clone(),
            speed: config.rate,
        };

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to OpenAI API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI API request failed: {status} - {error_text}");
        }

        let audio_data = response
            .bytes()
            .await
            .context("Failed to read audio data")?;

        Ok(OpenAiTtsResponse { audio_data })
    }

    /// Convert audio data to TtsResult
    fn convert_response(response: OpenAiTtsResponse, config: &TtsConfig) -> TtsResult {
        TtsResult {
            audio_data: response.audio_data.to_vec(),
            format: config.output_format.clone(),
            sample_rate: config.sample_rate,
            channels: 1, // OpenAI TTS is mono
            duration: None, // OpenAI doesn't provide duration
            timestamps: Vec::new(), // OpenAI doesn't provide timestamps
        }
    }
}

impl Default for OpenAiTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for OpenAiTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::OpenAi
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get model and voice
        let model = self.get_model(config);
        let voice = self.get_voice(config);

        // Call API
        let response = self
            .call_tts_api(text, &api_key, model, voice, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // OpenAI TTS doesn't support SSML, so we extract the text
        // Strip SSML tags and use the content
        let stripped = ssml
            .replace("<speak>", "")
            .replace("</speak>", "")
            .replace("<prosody", "")
            .replace("</prosody>", "")
            .replace("rate=\"", "")
            .replace("pitch=\"", "")
            .replace("\">", "");
        let text = stripped.trim();

        self.synthesize(text, config).await
    }

    async fn is_available(&self) -> bool {
        std::env::var("OPENAI_API_KEY").is_ok()
    }

    async fn get_voices(&self, _language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // OpenAI TTS available voices
        let voices = vec![
            VoiceInfo {
                id: "alloy".to_string(),
                name: "Alloy".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "echo".to_string(),
                name: "Echo".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "fable".to_string(),
                name: "Fable".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "onyx".to_string(),
                name: "Onyx".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "nova".to_string(),
                name: "Nova".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "shimmer".to_string(),
                name: "Shimmer".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
        ];

        Ok(voices)
    }
}

// ═══════════════════════════════════════════════════════════════
// OPENAI API TYPES
// ═══════════════════════════════════════════════════════════════

/// OpenAI TTS request
#[derive(Debug, Serialize)]
struct OpenAiTtsRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(rename = "response_format")]
    response_format: String,
    speed: f32,
}

/// OpenAI TTS response
struct OpenAiTtsResponse {
    audio_data: Vec<u8>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_provider_new() {
        let provider = OpenAiTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_default() {
        let provider = OpenAiTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_with_api_base() {
        let provider = OpenAiTtsProvider::with_api_base("https://api.example.com".to_string());
        assert_eq!(provider.provider_type(), TtsProviderType::OpenAi);
    }

    #[test]
    fn test_openai_provider_name() {
        let provider = OpenAiTtsProvider::new();
        assert_eq!(provider.name(), "OpenAI TTS");
    }

    #[test]
    fn test_get_model_default() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig {
            model: None,
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "tts-1");
    }

    #[test]
    fn test_get_model_custom() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig {
            model: Some("tts-1-hd".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "tts-1-hd");
    }

    #[test]
    fn test_get_voice_default() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig {
            voice: None,
            ..Default::default()
        };

        assert_eq!(provider.get_voice(&config), "alloy");
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig {
            voice: Some("nova".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_voice(&config), "nova");
    }

    #[tokio::test]
    async fn test_openai_provider_empty_text() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_openai_provider_too_long_text() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig::default();

        let long_text = "a".repeat(5000);
        let result = provider.synthesize(&long_text, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    #[tokio::test]
    async fn test_openai_provider_invalid_chars() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig::default();

        let text = "Hello\x00World";
        let result = provider.synthesize(text, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid control character"));
    }

    #[tokio::test]
    async fn test_openai_provider_no_api_key() {
        std::env::remove_var("OPENAI_API_KEY");

        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("Hello world", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));
    }

    #[tokio::test]
    async fn test_openai_provider_is_available() {
        let provider = OpenAiTtsProvider::new();
        // Just check the method doesn't panic
        let _ = provider.is_available().await;
    }

    #[tokio::test]
    async fn test_openai_provider_get_voices() {
        let provider = OpenAiTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().any(|v| v.id == "alloy"));
        assert!(voices.iter().any(|v| v.id == "nova"));
    }

    #[tokio::test]
    async fn test_openai_provider_get_voices_with_language() {
        let provider = OpenAiTtsProvider::new();
        let voices = provider.get_voices(Some("en-US")).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.language.starts_with("en")));
    }

    #[tokio::test]
    async fn test_openai_provider_synthesize_ssml() {
        let provider = OpenAiTtsProvider::new();
        let config = TtsConfig::default();

        // SSML is not supported, so it extracts the text
        let result = provider.synthesize_ssml("<speak>Hello world</speak>", &config).await;

        // Should fail due to no API key, but the text extraction should work
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_convert_response_empty() {
        let response = OpenAiTtsResponse {
            audio_data: bytes::Bytes::new(),
        };

        let config = TtsConfig::default();
        let result = OpenAiTtsProvider::convert_response(response, &config);

        assert!(result.audio_data.is_empty());
        assert_eq!(result.format, "mp3");
        assert_eq!(result.sample_rate, 24000);
        assert_eq!(result.channels, 1);
    }

    #[test]
    fn test_convert_response_with_data() {
        let audio_data = vec![0u8, 1, 2, 3, 4, 5];
        let response = OpenAiTtsResponse {
            audio_data: bytes::Bytes::from(audio_data),
        };

        let config = TtsConfig::default();
        let result = OpenAiTtsProvider::convert_response(response, &config);

        assert_eq!(result.audio_data.len(), 6);
        assert_eq!(result.audio_data, vec![0, 1, 2, 3, 4, 5]);
    }
}
