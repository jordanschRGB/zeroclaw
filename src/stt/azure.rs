// ═══════════════════════════════════════════════════════════════
// AZURE SPEECH SERVICES PROVIDER
// ═══════════════════════════════════════════════════════════════

#![allow(non_snake_case)]

use super::{SttConfig, SttProvider, SttProviderType, SttResult, audio_to_wav, validate_audio_data};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Azure Speech Services provider
pub struct AzureSttProvider {
    client: Client,
    api_base: String,
}

impl AzureSttProvider {
    /// Create a new Azure provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://*.cognitiveservices.azure.com".to_string(),
        }
    }

    /// Create a new provider with custom region
    pub fn with_region(region: String) -> Self {
        let api_base = format!("https://{region}.api.cognitive.microsoft.com");
        Self {
            client: Client::new(),
            api_base,
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
            .or_else(|| std::env::var("AZURE_SPEECH_KEY").ok())
            .context("Azure Speech key not found. Set AZURE_SPEECH_KEY environment variable or provide api_key in config.")
    }

    /// Get the region from environment
    fn get_region(&self) -> Result<String> {
        std::env::var("AZURE_SPEECH_REGION")
            .or_else(|_| std::env::var("AZURE_REGION"))
            .context("Azure region not found. Set AZURE_SPEECH_REGION environment variable.")
    }

    /// Get the API URL
    fn get_api_url(&self) -> Result<String> {
        if self.api_base.contains('*') {
            let region = self.get_region()?;
            Ok(format!("https://{region}.api.cognitive.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"))
        } else {
            Ok(format!("{}/speech/recognition/conversation/cognitiveservices/v1", self.api_base))
        }
    }

    /// Call the Azure Speech Services API
    async fn call_azure_api(
        &self,
        wav_data: Vec<u8>,
        api_key: &str,
        config: &SttConfig,
    ) -> Result<AzureResponse> {
        let url = self.get_api_url()?;

        let timeout = Duration::from_secs(config.timeout_secs);

        let mut query_params = vec![
            ("language", config.language.clone()),
            ("format", "detailed".to_string()),
        ];

        if config.enable_punctuation {
            query_params.push(("profanity", "raw".to_string()));
        } else {
            query_params.push(("profanity", "masked".to_string()));
        }

        let response = self
            .client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", api_key)
            .header("Content-Type", "audio/wav; codecs=audio/pcm; samplerate=16000")
            .query(&query_params)
            .body(wav_data)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to Azure API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure API request failed: {status} - {error_text}");
        }

        response
            .json::<AzureResponse>()
            .await
            .context("Failed to parse Azure API response")
    }

    /// Convert Azure response to SttResult
    fn convert_response(response: AzureResponse, _config: &SttConfig) -> SttResult {
        if response.RecognitionStatus != "Success" || response.NBest.is_empty() {
            return SttResult {
                text: String::new(),
                confidence: 0.0,
                language: response.PrimaryLanguage,
                alternatives: Vec::new(),
                words: Vec::new(),
            };
        }

        let best_alternative = &response.NBest[0];
        let text = best_alternative.Display.clone();
        let confidence = best_alternative.Confidence;

        let alternatives: Vec<super::SttAlternative> = response.NBest
            .iter()
            .skip(1)
            .map(|alt| super::SttAlternative {
                text: alt.Display.clone(),
                confidence: alt.Confidence,
            })
            .collect();

        // Azure doesn't provide word-level timestamps in all responses
        let words = Vec::new();

        SttResult {
            text,
            confidence,
            language: response.PrimaryLanguage,
            alternatives,
            words,
        }
    }
}

impl Default for AzureSttProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SttProvider for AzureSttProvider {
    fn provider_type(&self) -> SttProviderType {
        SttProviderType::Azure
    }

    async fn transcribe(&self, audio_data: &[f32], config: &SttConfig) -> Result<SttResult> {
        // Validate audio data
        validate_audio_data(audio_data)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Convert to WAV
        let wav_data = audio_to_wav(audio_data, config.sample_rate);

        // Call API
        let response = self
            .call_azure_api(wav_data, &api_key, config)
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

        // Call API with file data (already in WAV format)
        let response = self
            .call_azure_api(audio_data, &api_key, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("AZURE_SPEECH_KEY").is_ok()
    }
}

// ═══════════════════════════════════════════════════════════════
// AZURE API TYPES
// ═══════════════════════════════════════════════════════════════

/// Azure Speech-to-Text response
#[derive(Debug, Deserialize)]
struct AzureResponse {
    RecognitionStatus: String,
    DisplayText: Option<String>,
    Offset: Option<u64>,
    Duration: Option<u64>,
    PrimaryLanguage: Option<String>,
    #[serde(default)]
    Confidence: Option<f32>,
    #[serde(default)]
    NBest: Vec<AzureNBest>,
}

/// N-best alternative
#[derive(Debug, Deserialize)]
struct AzureNBest {
    Lexical: String,
    ITN: String,
    MaskedITN: String,
    Display: String,
    Confidence: f32,
    LexicalScore: Option<i32>,
    ITNScore: Option<i32>,
    MaskedITNScore: Option<i32>,
    DisplayScore: Option<i32>,
    Words: Vec<AzureWord>,
}

/// Word info
#[derive(Debug, Deserialize)]
struct AzureWord {
    Word: String,
    Offset: u64,
    Duration: u64,
    Confidence: Option<f32>,
    PronunciationAccuracy: Option<String>,
    ErrorType: Option<String>,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_provider_new() {
        let provider = AzureSttProvider::new();
        assert_eq!(provider.provider_type(), SttProviderType::Azure);
    }

    #[test]
    fn test_azure_provider_default() {
        let provider = AzureSttProvider::default();
        assert_eq!(provider.provider_type(), SttProviderType::Azure);
    }

    #[test]
    fn test_azure_provider_with_region() {
        let provider = AzureSttProvider::with_region("westus".to_string());
        assert_eq!(provider.provider_type(), SttProviderType::Azure);
        assert!(provider.api_base.contains("westus"));
    }

    #[test]
    fn test_azure_provider_name() {
        let provider = AzureSttProvider::new();
        assert_eq!(provider.name(), "Azure Speech Services");
    }

    #[test]
    fn test_azure_response_success() {
        let response = AzureResponse {
            RecognitionStatus: "Success".to_string(),
            DisplayText: Some("hello world".to_string()),
            Offset: Some(0),
            Duration: Some(1000000),
            PrimaryLanguage: Some("en-US".to_string()),
            Confidence: Some(0.95),
            NBest: vec![
                AzureNBest {
                    Lexical: "hello world".to_string(),
                    ITN: "hello world".to_string(),
                    MaskedITN: "hello world".to_string(),
                    Display: "hello world".to_string(),
                    Confidence: 0.95,
                    LexicalScore: None,
                    ITNScore: None,
                    MaskedITNScore: None,
                    DisplayScore: None,
                    Words: vec![],
                },
            ],
        };

        let result = AzureSttProvider::convert_response(response, &SttConfig::default());
        assert_eq!(result.text, "hello world");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.language, Some("en-US".to_string()));
    }

    #[test]
    fn test_azure_response_failure() {
        let response = AzureResponse {
            RecognitionStatus: "NoMatch".to_string(),
            DisplayText: None,
            Offset: Some(0),
            Duration: Some(1000000),
            PrimaryLanguage: None,
            Confidence: None,
            NBest: vec![],
        };

        let result = AzureSttProvider::convert_response(response, &SttConfig::default());
        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[tokio::test]
    async fn test_azure_provider_empty_audio() {
        let provider = AzureSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_azure_provider_invalid_audio() {
        let provider = AzureSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.5, f32::NAN, 0.3];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_azure_provider_no_region_wildcard() {
        let provider = AzureSttProvider::new();
        assert!(provider.api_base.contains('*'));
    }

    #[tokio::test]
    async fn test_azure_provider_too_quiet_audio() {
        let provider = AzureSttProvider::new();
        let config = SttConfig::default();

        let audio: Vec<f32> = vec![0.0001; 1000];
        let result = provider.transcribe(&audio, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too quiet"));
    }

    #[tokio::test]
    async fn test_azure_provider_is_available() {
        let provider = AzureSttProvider::new();
        // Just check the method doesn't panic
        let _ = provider.is_available().await;
    }

    #[test]
    fn test_azure_response_with_nbest() {
        let response = AzureResponse {
            RecognitionStatus: "Success".to_string(),
            DisplayText: Some("hello world".to_string()),
            Offset: Some(0),
            Duration: Some(1000000),
            PrimaryLanguage: Some("en-US".to_string()),
            Confidence: Some(0.95),
            NBest: vec![
                AzureNBest {
                    Lexical: "hello world".to_string(),
                    ITN: "hello world".to_string(),
                    MaskedITN: "hello world".to_string(),
                    Display: "Hello world.".to_string(),
                    Confidence: 0.95,
                    LexicalScore: None,
                    ITNScore: None,
                    MaskedITNScore: None,
                    DisplayScore: None,
                    Words: vec![],
                },
            ],
        };

        let result = AzureSttProvider::convert_response(response, &SttConfig::default());
        assert_eq!(result.text, "Hello world.");
    }
}
