// ═══════════════════════════════════════════════════════════════
// GOOGLE CLOUD TEXT-TO-SPEECH PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceInfo, VoiceGender, validate_text};
use base64::Engine;
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Google Cloud TTS provider
pub struct GoogleTtsProvider {
    client: Client,
    api_base: String,
}

impl GoogleTtsProvider {
    /// Create a new Google TTS provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://texttospeech.googleapis.com/v1".to_string(),
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
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .context("Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the voice to use
    fn get_voice(&self, config: &TtsConfig) -> String {
        config
            .voice
            .clone()
            .unwrap_or_else(|| format!("{}-Neural2", config.language.replace('-', "-")))
    }

    /// Call the Google TTS API
    async fn call_tts_api(
        &self,
        text: &str,
        api_key: &str,
        voice: &str,
        config: &TtsConfig,
    ) -> Result<GoogleTtsResponse> {
        let url = format!("{}:synthesize?key={}", self.api_base, api_key);

        let request = GoogleTtsRequest {
            input: GoogleTtsInput {
                text: text.to_string(),
            },
            voice: GoogleTtsVoice {
                language_code: config.language.clone(),
                name: Some(voice.to_string()),
                ssml_gender: None,
            },
            audio_config: GoogleTtsAudioConfig {
                audio_encoding: match config.output_format.as_str() {
                    "mp3" => "MP3".to_string(),
                    "wav" => "LINEAR16".to_string(),
                    "pcm" => "LINEAR16".to_string(),
                    _ => "MP3".to_string(),
                },
                speaking_rate: (config.rate * 100.0) as i32,
                pitch: config.pitch as i32,
                sample_rate_hertz: config.sample_rate as i32,
            },
        };

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to Google TTS API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Google TTS API request failed: {status} - {error_text}");
        }

        response
            .json::<GoogleTtsResponse>()
            .await
            .context("Failed to parse Google TTS API response")
    }

    /// Convert response to TtsResult
    fn convert_response(response: GoogleTtsResponse, config: &TtsConfig) -> TtsResult {
        let audio_data = base64::engine::general_purpose::STANDARD.decode(&response.audio_content).unwrap_or_default();

        TtsResult {
            audio_data,
            format: config.output_format.clone(),
            sample_rate: config.sample_rate,
            channels: 1,
            duration: Some(response.time_offset_secs.unwrap_or(0.0)),
            timestamps: Vec::new(),
        }
    }
}

impl Default for GoogleTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for GoogleTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::Google
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get voice
        let voice = self.get_voice(config);

        // Call API
        let response = self
            .call_tts_api(text, &api_key, &voice, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Extract text from SSML
        let stripped = ssml
            .replace("<speak>", "")
            .replace("</speak>", "")
            .replace("<prosody>", "")
            .replace("</prosody>", "")
            .replace("<break time=", "") // Keep break markers
            .replace("\" />", "");
        let text = stripped.trim();

        self.synthesize(text, config).await
    }

    async fn is_available(&self) -> bool {
        std::env::var("GOOGLE_API_KEY").is_ok()
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // Return a subset of popular Google TTS voices
        let mut voices = vec![
            VoiceInfo {
                id: "en-US-Neural2".to_string(),
                name: "US English Neural 2".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en-US-Studio-O".to_string(),
                name: "US English Studio O".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(false),
            },
            VoiceInfo {
                id: "en-US-Standard".to_string(),
                name: "US English Standard".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Neutral,
                sample_rate: Some(16000),
                neural: Some(false),
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
// GOOGLE API TYPES
// ═══════════════════════════════════════════════════════════════

/// Google TTS request
#[derive(Debug, Serialize)]
struct GoogleTtsRequest {
    input: GoogleTtsInput,
    voice: GoogleTtsVoice,
    audio_config: GoogleTtsAudioConfig,
}

/// Input text
#[derive(Debug, Serialize)]
struct GoogleTtsInput {
    text: String,
}

/// Voice configuration
#[derive(Debug, Serialize)]
struct GoogleTtsVoice {
    language_code: String,
    name: Option<String>,
    ssml_gender: Option<String>,
}

/// Audio configuration
#[derive(Debug, Serialize)]
struct GoogleTtsAudioConfig {
    audio_encoding: String,
    speaking_rate: i32,
    pitch: i32,
    sample_rate_hertz: i32,
}

/// Google TTS response
#[derive(Debug, Deserialize)]
struct GoogleTtsResponse {
    audio_content: String,
    time_offset_secs: Option<f32>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_google_provider_new() {
        let provider = GoogleTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::Google);
    }

    #[test]
    fn test_google_provider_default() {
        let provider = GoogleTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::Google);
    }

    #[test]
    fn test_google_provider_name() {
        let provider = GoogleTtsProvider::new();
        assert_eq!(provider.name(), "Google Cloud Text-to-Speech");
    }

    #[test]
    fn test_get_voice_default() {
        let provider = GoogleTtsProvider::new();
        let config = TtsConfig::default();

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "en-US-Neural2");
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = GoogleTtsProvider::new();
        let config = TtsConfig {
            voice: Some("en-US-Standard".to_string()),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "en-US-Standard");
    }

    #[tokio::test]
    async fn test_google_provider_empty_text() {
        let provider = GoogleTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_google_provider_no_api_key() {
        std::env::remove_var("GOOGLE_API_KEY");

        let provider = GoogleTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("Hello world", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));
    }

    #[tokio::test]
    async fn test_google_provider_is_available() {
        let provider = GoogleTtsProvider::new();
        let _ = provider.is_available().await;
    }

    #[tokio::test]
    async fn test_google_provider_get_voices() {
        let provider = GoogleTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_google_provider_get_voices_with_language() {
        let provider = GoogleTtsProvider::new();
        let voices = provider.get_voices(Some("en-US")).await.unwrap();

        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_google_provider_get_voices_no_match() {
        let provider = GoogleTtsProvider::new();
        let voices = provider.get_voices(Some("zh-CN")).await.unwrap();

        assert!(voices.is_empty());
    }

    #[test]
    fn test_convert_response_empty() {
        let response = GoogleTtsResponse {
            audio_content: String::new(),
            time_offset_secs: Some(1.5),
        };

        let config = TtsConfig::default();
        let result = GoogleTtsProvider::convert_response(response, &config);

        assert!(result.audio_data.is_empty());
        assert_eq!(result.duration, Some(1.5));
    }

    #[test]
    fn test_convert_response_with_data() {
        // "SGVsbG8gd29ybGQ=" is "Hello world" in base64
        let response = GoogleTtsResponse {
            audio_content: "SGVsbG8gd29ybGQ=".to_string(),
            time_offset_secs: Some(1.0),
        };

        let config = TtsConfig::default();
        let result = GoogleTtsProvider::convert_response(response, &config);

        assert!(!result.audio_data.is_empty());
        assert_eq!(result.duration, Some(1.0));
    }
}
