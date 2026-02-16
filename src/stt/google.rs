// ═══════════════════════════════════════════════════════════════
// GOOGLE CLOUD SPEECH-TO-TEXT PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{SttConfig, SttProvider, SttProviderType, SttResult, SttAlternative, SttWord, audio_to_wav, validate_audio_data};
use base64::Engine;
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Google Cloud Speech-to-Text provider
pub struct GoogleSttProvider {
    client: Client,
    api_base: String,
}

impl GoogleSttProvider {
    /// Create a new Google provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://speech.googleapis.com/v1".to_string(),
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
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .context("Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key in config.")
    }

    /// Get the model to use
    fn get_model(&self, config: &SttConfig) -> String {
        config
            .model
            .clone()
            .unwrap_or_else(|| "latest_long".to_string())
    }

    /// Call the Google Speech-to-Text API
    async fn call_google_api(
        &self,
        wav_data: Vec<u8>,
        api_key: &str,
        model: &str,
        config: &SttConfig,
    ) -> Result<GoogleResponse> {
        let url = format!(
            "{}/speech:recognize?key={}",
            self.api_base, api_key
        );

        // Build request
        let request = GoogleRequest {
            config: GoogleRecognitionConfig {
                encoding: Some("LINEAR16".to_string()),
                sample_rate_hertz: Some(config.sample_rate as i32),
                language_code: config.language.clone(),
                model: Some(model.to_string()),
                enable_automatic_punctuation: Some(config.enable_punctuation),
                profanity_filter: config.filter_profanity,
                audio_channel_count: Some(1),
                enable_word_time_offsets: Some(true),
                max_alternatives: Some(3),
            },
            audio: GoogleRecognitionAudio {
                content: Some(base64::engine::general_purpose::STANDARD.encode(&wav_data)),
                uri: None,
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
            .context("Failed to send request to Google API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Google API request failed: {status} - {error_text}");
        }

        response
            .json::<GoogleResponse>()
            .await
            .context("Failed to parse Google API response")
    }

    /// Convert Google response to SttResult
    fn convert_response(response: GoogleResponse, _config: &SttConfig) -> SttResult {
        if response.results.is_empty() {
            return SttResult {
                text: String::new(),
                confidence: 0.0,
                language: None,
                alternatives: Vec::new(),
                words: Vec::new(),
            };
        }

        let best_result = &response.results[0];
        let best_alternative = &best_result.alternatives[0];

        let text = best_alternative.transcript.clone();
        let confidence = best_alternative.confidence.unwrap_or(0.0);

        let alternatives: Vec<SttAlternative> = best_result
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
                start_time: w.start_time.clone().unwrap_or_default().as_seconds(),
                end_time: w.end_time.clone().unwrap_or_default().as_seconds(),
                confidence: 0.95, // Google doesn't provide word-level confidence
            })
            .collect();

        SttResult {
            text,
            confidence,
            language: None, // Could be extracted from response
            alternatives,
            words,
        }
    }
}

impl Default for GoogleSttProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for GoogleSttProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::Google
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
            .call_google_api(wav_data, &api_key, &model, config)
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
            .call_google_api(audio_data, &api_key, &model, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("GOOGLE_API_KEY").is_ok()
    }
}

// ═══════════════════════════════════════════════════════════════
// GOOGLE API TYPES
// ═══════════════════════════════════════════════════════════════

/// Google Speech-to-Text request
#[derive(Debug, Serialize)]
struct GoogleRequest {
    config: GoogleRecognitionConfig,
    audio: GoogleRecognitionAudio,
}

/// Recognition config
#[derive(Debug, Serialize)]
struct GoogleRecognitionConfig {
    encoding: Option<String>,
    sample_rate_hertz: Option<i32>,
    language_code: String,
    model: Option<String>,
    enable_automatic_punctuation: Option<bool>,
    profanity_filter: bool,
    audio_channel_count: Option<i32>,
    enable_word_time_offsets: Option<bool>,
    max_alternatives: Option<i32>,
}

/// Recognition audio
#[derive(Debug, Serialize)]
struct GoogleRecognitionAudio {
    content: Option<String>,
    uri: Option<String>,
}

/// Google Speech-to-Text response
#[derive(Debug, Deserialize)]
struct GoogleResponse {
    results: Vec<GoogleSpeechResult>,
}

/// Speech result
#[derive(Debug, Deserialize)]
struct GoogleSpeechResult {
    alternatives: Vec<GoogleSpeechAlternative>,
}

/// Speech alternative
#[derive(Debug, Deserialize)]
struct GoogleSpeechAlternative {
    transcript: String,
    confidence: Option<f32>,
    words: Vec<GoogleWord>,
}

/// Word info
#[derive(Debug, Deserialize)]
struct GoogleWord {
    word: String,
    start_time: Option<GoogleDuration>,
    end_time: Option<GoogleDuration>,
}

/// Duration
#[derive(Debug, Deserialize, Clone)]
struct GoogleDuration {
    seconds: i64,
    nanos: i32,
}

impl GoogleDuration {
    fn as_seconds(&self) -> f32 {
        self.seconds as f32 + (self.nanos as f32 / 1_000_000_000.0)
    }

    fn as_default() -> Self {
        Self {
            seconds: 0,
            nanos: 0,
        }
    }
}

impl Default for GoogleDuration {
    fn default() -> Self {
        Self::as_default()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_google_provider_new() {
        let provider = GoogleSttProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::Google);
    }

    #[test]
    fn test_google_provider_default() {
        let provider = GoogleSttProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::Google);
    }

    #[test]
    fn test_google_provider_name() {
        let provider = GoogleSttProvider::new();
        assert_eq!(provider.name(), "Google Cloud Speech-to-Text");
    }

    #[test]
    fn test_google_response_empty() {
        let response = GoogleResponse {
            results: Vec::new(),
        };

        let result = GoogleSttProvider::convert_response(response, &SttConfig::default());
        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_google_response_with_transcript() {
        let response = GoogleResponse {
            results: vec![
                GoogleSpeechResult {
                    alternatives: vec![
                        GoogleSpeechAlternative {
                            transcript: "hello world".to_string(),
                            confidence: Some(0.95),
                            words: vec![
                                GoogleWord {
                                    word: "hello".to_string(),
                                    start_time: Some(GoogleDuration { seconds: 0, nanos: 0 }),
                                    end_time: Some(GoogleDuration { seconds: 0, nanos: 500_000_000 }),
                                },
                                GoogleWord {
                                    word: "world".to_string(),
                                    start_time: Some(GoogleDuration { seconds: 0, nanos: 500_000_000 }),
                                    end_time: Some(GoogleDuration { seconds: 1, nanos: 0 }),
                                },
                            ],
                        },
                        GoogleSpeechAlternative {
                            transcript: "hello word".to_string(),
                            confidence: Some(0.85),
                            words: vec![],
                        },
                    ],
                },
            ],
        };

        let result = GoogleSttProvider::convert_response(response, &SttConfig::default());
        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.alternatives.len(), 1);
        assert_eq!(result.alternatives[0].text, "hello word");
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[0].word, "hello");
        assert_eq!(result.words[1].word, "world");
    }

    #[test]
    fn test_google_duration_as_seconds() {
        let duration = GoogleDuration {
            seconds: 1,
            nanos: 500_000_000,
        };
        assert_eq!(duration.as_seconds(), 1.5);
    }

    #[test]
    fn test_google_duration_zero() {
        let duration = GoogleDuration {
            seconds: 0,
            nanos: 0,
        };
        assert_eq!(duration.as_seconds(), 0.0);
    }

    #[tokio::test]
    async fn test_google_provider_empty_audio() {
        let provider = GoogleSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_google_provider_invalid_audio() {
        let provider = GoogleSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.5, f32::NAN, 0.3];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_google_provider_no_api_key() {
        std::env::remove_var("GOOGLE_API_KEY");

        let provider = GoogleSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.1; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not found"));
    }

    #[test]
    fn test_get_model_default() {
        let provider = GoogleSttProvider::new();
        let config = SttConfig {
            model: None,
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "latest_long");
    }

    #[test]
    fn test_get_model_custom() {
        let provider = GoogleSttProvider::new();
        let config = SttConfig {
            model: Some("latest_short".to_string()),
            ..Default::default()
        };

        assert_eq!(provider.get_model(&config), "latest_short");
    }

    #[tokio::test]
    async fn test_google_provider_too_quiet_audio() {
        let provider = GoogleSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.0001; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too quiet"));
    }

    #[tokio::test]
    async fn test_google_provider_is_available() {
        let provider = GoogleSttProvider::new();
        // Just check the method doesn't panic
        let _ = provider.is_available().await;
    }
}
