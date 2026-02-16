// ═══════════════════════════════════════════════════════════════
// ELEVENLABS TTS PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceInfo, VoiceGender, validate_text};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;
use std::time::Duration;

/// ElevenLabs TTS provider
pub struct ElevenLabsTtsProvider {
    client: Client,
    api_base: String,
}

impl ElevenLabsTtsProvider {
    /// Create a new ElevenLabs provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://api.elevenlabs.io/v1".to_string(),
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
            .or_else(|| std::env::var("ELEVENLABS_API_KEY").ok())
            .context("ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the voice ID to use
    fn get_voice(&self, config: &TtsConfig) -> String {
        config
            .voice
            .clone()
            .unwrap_or_else(|| "21m00Tcm4TlvDq8ikWAM".to_string()) // Rachel (default)
    }

    /// Get the model to use
    fn get_model(&self, config: &TtsConfig) -> &str {
        config
            .model
            .as_deref()
            .unwrap_or("eleven_multilingual_v2")
    }

    /// Call the ElevenLabs TTS API
    async fn call_tts_api(
        &self,
        text: &str,
        api_key: &str,
        voice_id: &str,
        model: &str,
        config: &TtsConfig,
    ) -> Result<ElevenLabsTtsResponse> {
        let url = format!("{}/text-to-speech/{}", self.api_base, voice_id);

        let mut request = ElevenLabsRequest {
            text: text.to_string(),
            model_id: model.to_string(),
            voice_settings: Some(ElevenLabsVoiceSettings {
                stability: 0.5,
                similarity_boost: 0.75,
            }),
        };

        // Adjust for rate and pitch if provided
        if config.rate != 1.0 || config.pitch != 0.0 {
            // Convert rate (0.1-2.0) to stability (0.0-1.0)
            // Higher rate = lower stability
            let stability = (1.0 - (config.rate - 1.0).abs().min(1.0)).max(0.0);

            if let Some(ref mut settings) = request.voice_settings {
                settings.stability = stability;
            }
        }

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("xi-api-key", api_key)
            .header("accept", "audio/mpeg")
            .header("Content-Type", "application/json")
            .json(&request)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to ElevenLabs API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("ElevenLabs API request failed: {status} - {error_text}");
        }

        let audio_data = response
            .bytes()
            .await
            .context("Failed to read audio data")?;

        Ok(ElevenLabsTtsResponse { audio_data: audio_data.to_vec() })
    }

    /// Convert audio data to TtsResult
    fn convert_response(response: ElevenLabsTtsResponse, config: &TtsConfig) -> TtsResult {
        TtsResult {
            audio_data: response.audio_data.to_vec(),
            format: "mp3".to_string(), // ElevenLabs returns MP3
            sample_rate: config.sample_rate,
            channels: 1, // ElevenLabs is mono
            duration: None,
            timestamps: Vec::new(),
        }
    }
}

impl Default for ElevenLabsTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for ElevenLabsTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::ElevenLabs
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get voice and model
        let voice_id = self.get_voice(config);
        let model = self.get_model(config);

        // Call API
        let response = self
            .call_tts_api(text, &api_key, &voice_id, model, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // ElevenLabs supports SSML with proper setup
        // For now, extract text like OpenAI
        let stripped = ssml
            .replace("<speak>", "")
            .replace("</speak>", "")
            .replace("<prosody>", "")
            .replace("</prosody>", "");
        let text = stripped.trim();

        self.synthesize(text, config).await
    }

    async fn is_available(&self) -> bool {
        std::env::var("ELEVENLABS_API_KEY").is_ok()
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // ElevenLabs has many voices, return a subset of popular ones
        let mut voices = vec![
            VoiceInfo {
                id: "21m00Tcm4TlvDq8ikWAM".to_string(),
                name: "Rachel".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "AZnzlk1XvdvUeB6XmltgUK".to_string(),
                name: "Domi".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "EXAVITQu4vr4Ljzng6ftEu".to_string(),
                name: "Fin".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(true),
            },
        ];

        // Filter by language if specified
        if let Some(lang) = language {
            voices = voices
                .into_iter()
                .filter(|v| v.language == lang)
                .collect();
        }

        Ok(voices)
    }
}

// ═══════════════════════════════════════════════════════════════
// ELEVENLABS API TYPES
// ═══════════════════════════════════════════════════════════════

/// ElevenLabs TTS request
#[derive(Debug, Serialize)]
struct ElevenLabsRequest {
    text: String,
    model_id: String,
    voice_settings: Option<ElevenLabsVoiceSettings>,
}

/// ElevenLabs voice settings
#[derive(Debug, Serialize)]
struct ElevenLabsVoiceSettings {
    stability: f32,
    similarity_boost: f32,
}

/// ElevenLabs TTS response
struct ElevenLabsTtsResponse {
    audio_data: Vec<u8>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elevenlabs_provider_new() {
        let provider = ElevenLabsTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::ElevenLabs);
    }

    #[test]
    fn test_elevenlabs_provider_default() {
        let provider = ElevenLabsTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::ElevenLabs);
    }

    #[test]
    fn test_elevenlabs_provider_name() {
        let provider = ElevenLabsTtsProvider::new();
        assert_eq!(provider.name(), "ElevenLabs");
    }

    #[test]
    fn test_get_model_default() {
        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig {
            model: None,
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "eleven_multilingual_v2");
    }

    #[test]
    fn test_get_model_custom() {
        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig {
            model: Some("eleven_monolingual_v1".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "eleven_monolingual_v1");
    }

    #[test]
    fn test_get_voice_default() {
        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig {
            voice: None,
            ..Default::default()
        };

        assert_eq!(provider.get_voice(&config), "21m00Tcm4TlvDq8ikWAM");
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig {
            voice: Some("AZnzlk1XvdvUeB6XmltgUK".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_voice(&config), "AZnzlk1XvdvUeB6XmltgUK");
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_empty_text() {
        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_no_api_key() {
        std::env::remove_var("ELEVENLABS_API_KEY");

        let provider = ElevenLabsTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("Hello world", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_is_available() {
        let provider = ElevenLabsTtsProvider::new();
        // Just check the method doesn't panic
        let _ = provider.is_available().await;
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_get_voices() {
        let provider = ElevenLabsTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().any(|v| v.id == "21m00Tcm4TlvDq8ikWAM"));
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_get_voices_with_language() {
        let provider = ElevenLabsTtsProvider::new();
        let voices = provider.get_voices(Some("en-US")).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.language == "en-US"));
    }

    #[tokio::test]
    async fn test_elevenlabs_provider_get_voices_no_match() {
        let provider = ElevenLabsTtsProvider::new();
        let voices = provider.get_voices(Some("zh-CN")).await.unwrap();

        // Should return empty list if no voices match
        assert!(voices.is_empty());
    }

    #[test]
    fn test_convert_response_empty() {
        let response = ElevenLabsTtsResponse {
            audio_data: vec![],
        };

        let config = TtsConfig::default();
        let result = ElevenLabsTtsProvider::convert_response(response, &config);

        assert!(result.audio_data.is_empty());
        assert_eq!(result.format, "mp3");
    }

    #[test]
    fn test_convert_response_with_data() {
        let audio_data = vec![0u8, 1, 2, 3, 4, 5];
        let response = ElevenLabsTtsResponse {
            audio_data: audio_data.clone(),
        };

        let config = TtsConfig::default();
        let result = ElevenLabsTtsProvider::convert_response(response, &config);

        assert_eq!(result.audio_data.len(), 6);
    }
}
