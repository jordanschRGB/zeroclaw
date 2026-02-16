// ═══════════════════════════════════════════════════════════════
// DEEPGRAM SPEECH-TO-TEXT PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{SttConfig, SttProvider, SttProviderType, SttResult, SttAlternative, SttWord, audio_to_wav, validate_audio_data};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Deepgram STT provider
pub struct DeepgramSttProvider {
    client: Client,
    api_base: String,
}

impl DeepgramSttProvider {
    /// Create a new Deepgram provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://api.deepgram.com/v1".to_string(),
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
            .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
            .context("Deepgram API key not found. Set DEEPGRAM_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the model to use
    fn get_model(&self, config: &SttConfig) -> String {
        config
            .model
            .clone()
            .unwrap_or_else(|| "nova-2".to_string())
    }

    /// Call the Deepgram API
    async fn call_deepgram_api(
        &self,
        wav_data: Vec<u8>,
        api_key: &str,
        model: &str,
        config: &SttConfig,
    ) -> Result<DeepgramResponse> {
        let url = format!(
            "{}/listen?model={}&language={}&punctuate={}&profanity_filter={}&smart_format=true&utterances=true",
            self.api_base,
            model,
            config.language.to_lowercase().replace('-', "_"),
            config.enable_punctuation,
            !config.filter_profanity
        );

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Token {api_key}"))
            .header("Content-Type", "audio/wav")
            .body(wav_data)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to Deepgram API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Deepgram API request failed: {status} - {error_text}");
        }

        response
            .json::<DeepgramResponse>()
            .await
            .context("Failed to parse Deepgram API response")
    }

    /// Convert Deepgram response to SttResult
    fn convert_response(response: DeepgramResponse, _config: &SttConfig) -> SttResult {
        if response.results.channels.is_empty()
            || response.results.channels[0].alternatives.is_empty()
        {
            return SttResult {
                text: String::new(),
                confidence: 0.0,
                language: None,
                alternatives: Vec::new(),
                words: Vec::new(),
            };
        }

        let channel = &response.results.channels[0];
        let best_alternative = &channel.alternatives[0];

        let text = best_alternative.transcript.clone();
        let confidence = best_alternative.confidence.unwrap_or(0.0);

        let alternatives: Vec<SttAlternative> = channel
            .alternatives
            .iter()
            .skip(1)
            .map(|alt| SttAlternative {
                text: alt.transcript.clone(),
                confidence: alt.confidence.unwrap_or(0.0),
            })
            .collect();

        let words: Vec<SttWord> = best_alternative
            .words
            .iter()
            .map(|w| SttWord {
                word: w.word.clone(),
                start_time: w.start,
                end_time: w.end,
                confidence: w.confidence.unwrap_or(0.95),
            })
            .collect();

        SttResult {
            text,
            confidence,
            language: None, // Deepgram doesn't return language in response
            alternatives,
            words,
        }
    }
}

impl Default for DeepgramSttProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for DeepgramSttProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::Deepgram
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
            .call_deepgram_api(wav_data, &api_key, &model, config)
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

        // Call API with file data (already in WAV format)
        let response = self
            .call_deepgram_api(audio_data, &api_key, &model, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("DEEPGRAM_API_KEY").is_ok()
    }
}

// ═══════════════════════════════════════════════════════════════
// DEEPGRAM API TYPES
// ═══════════════════════════════════════════════════════════════

/// Deepgram response
#[derive(Debug, Deserialize)]
struct DeepgramResponse {
    results: DeepgramResults,
}

/// Deepgram results
#[derive(Debug, Deserialize)]
struct DeepgramResults {
    channels: Vec<DeepgramChannel>,
}

/// Deepgram channel
#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    alternatives: Vec<DeepgramAlternative>,
}

/// Deepgram alternative
#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    transcript: String,
    confidence: Option<f32>,
    words: Vec<DeepgramWord>,
}

/// Deepgram word
#[derive(Debug, Deserialize)]
struct DeepgramWord {
    word: String,
    start: f32,
    end: f32,
    confidence: Option<f32>,
    punctuated_word: Option<String>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepgram_provider_new() {
        let provider = DeepgramSttProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::Deepgram);
    }

    #[test]
    fn test_deepgram_provider_default() {
        let provider = DeepgramSttProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::Deepgram);
    }

    #[test]
    fn test_deepgram_provider_name() {
        let provider = DeepgramSttProvider::new();
        assert_eq!(provider.name(), "Deepgram");
    }

    #[test]
    fn test_deepgram_response_with_transcript() {
        let json = r#"{
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "confidence": 0.95,
                                "words": [
                                    {
                                        "word": "hello",
                                        "start": 0.0,
                                        "end": 0.5,
                                        "confidence": 0.98,
                                        "punctuated_word": "Hello"
                                    },
                                    {
                                        "word": "world",
                                        "start": 0.5,
                                        "end": 1.0,
                                        "confidence": 0.92,
                                        "punctuated_word": "world"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }"#;

        let response: DeepgramResponse = serde_json::from_str(json).unwrap();
        let result = DeepgramSttProvider::convert_response(response, &SttConfig::default());

        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].word, "hello");
        assert_eq!(result.words[1].word, "world");
    }

    #[test]
    fn test_deepgram_response_empty() {
        let response = DeepgramResponse {
            results: DeepgramResults {
                channels: vec![],
            },
        };

        let result = DeepgramSttProvider::convert_response(response, &SttConfig::default());
        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_deepgram_response_with_alternatives() {
        let json = r#"{
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "confidence": 0.95,
                                "words": []
                            },
                            {
                                "transcript": "hello word",
                                "confidence": 0.85,
                                "words": []
                            }
                        ]
                    }
                ]
            }
        }"#;

        let response: DeepgramResponse = serde_json::from_str(json).unwrap();
        let result = DeepgramSttProvider::convert_response(response, &SttConfig::default());

        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.alternatives.len(), 1);
        assert_eq!(result.alternatives[0].text, "hello word");
        assert_eq!(result.alternatives[0].confidence, 0.85);
    }

    #[tokio::test]
    async fn test_deepgram_provider_empty_audio() {
        let provider = DeepgramSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_deepgram_provider_invalid_audio() {
        let provider = DeepgramSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.5, f32::NAN, 0.3];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deepgram_provider_no_api_key() {
        std::env::remove_var("DEEPGRAM_API_KEY");

        let provider = DeepgramSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.1; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));
    }

    #[test]
    fn test_get_model_default() {
        let provider = DeepgramSttProvider::new();
        let config = SttConfig {
            model: None,
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "nova-2");
    }

    #[test]
    fn test_get_model_custom() {
        let provider = DeepgramSttProvider::new();
        let config = SttConfig {
            model: Some("nova-3".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "nova-3");
    }

    #[tokio::test]
    async fn test_deepgram_provider_too_quiet_audio() {
        let provider = DeepgramSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.0001; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too quiet"));
    }

    #[tokio::test]
    async fn test_deepgram_provider_is_available() {
        let provider = DeepgramSttProvider::new();
        // Just check the method doesn't panic
        let _ = provider.is_available().await;
    }
}
