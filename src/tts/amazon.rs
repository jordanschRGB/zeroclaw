// ═══════════════════════════════════════════════════════════════
// AMAZON POLLY TEXT-TO-SPEECH PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceInfo, VoiceGender, text_to_ssml, validate_text};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::Serialize;
use std::time::Duration;
use hmac::{Hmac, Mac};
use sha2::{Sha256, Digest};

type HmacSha256 = Hmac<Sha256>;

/// Amazon Polly TTS provider
pub struct AmazonTtsProvider {
    client: Client,
    api_base: String,
    region: String,
}

impl AmazonTtsProvider {
    /// Create a new Amazon Polly provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://polly.amazonaws.com".to_string(),
            region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
        }
    }

    /// Create a new provider with custom region
    pub fn with_region(region: String) -> Self {
        Self {
            client: Client::new(),
            api_base: "https://polly.amazonaws.com".to_string(),
            region,
        }
    }

    /// Create a new provider with custom API base
    pub fn with_api_base(api_base: String) -> Self {
        Self {
            client: Client::new(),
            api_base,
            region: "us-east-1".to_string(),
        }
    }

    /// Get the AWS credentials from environment or config
    fn get_credentials(&self, config: &TtsConfig) -> Result<(String, String)> {
        let access_key = config
            .api_key
            .as_ref()
            .cloned()
            .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
            .context("AWS access key not found. Set AWS_ACCESS_KEY_ID environment variable or provide api_key in config.")?;

        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .or_else(|_| std::env::var("AWS_SECRET_KEY"))
            .context("AWS secret key not found. Set AWS_SECRET_ACCESS_KEY environment variable.")?;

        Ok((access_key, secret_key))
    }

    /// Get the voice to use
    fn get_voice(&self, config: &TtsConfig) -> String {
        config
            .voice
            .clone()
            .unwrap_or_else(|| "Joanna".to_string())
    }

    /// Get the engine to use (standard or neural)
    fn get_engine(&self, config: &TtsConfig) -> String {
        // Check if the voice is a neural voice
        let neural_voices = vec![
            "Ada", "Amy", "Aria", "Arthur", "Ayanda", "Bianca", "Brian",
            "Camila", "Carol", "Catherine", "Celine", "Chantal", "Cristiano",
            "Daniel", "Elin", "Emma", "Gabrielle", "Hans", "Ivy", "Jorge",
            "Kendra", "Kevin", "Kajal", "Karl", "Kendra", "Kimberly", "Lea",
            "Liam", "Liv", "Matthew", "Mia", "Miguel", "Niamh", "Olivia",
            "Penelope", "Raveena", "Ruth", "Stephen", "Takumi", "Vicki", "Vitoria",
        ];

        let voice = self.get_voice(config);
        let voice_name = voice.split('-').next().unwrap_or(&voice);

        if neural_voices.contains(&voice_name) {
            "neural".to_string()
        } else {
            "standard".to_string()
        }
    }

    /// Generate AWS Signature Version 4
    fn sign_request(
        &self,
        access_key: &str,
        secret_key: &str,
        method: &str,
        service: &str,
        region: &str,
        payload: &[u8],
    ) -> Result<String> {
        let host = "polly.amazonaws.com";
        let endpoint = "/";
        let algorithm = "AWS4-HMAC-SHA256";
        let now = chrono::Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();

        // Step 1: Create canonical request
        let canonical_uri = endpoint;
        let canonical_querystring = "";
        let canonical_headers = format!(
            "host:{}\nx-amz-content-sha256:{}\nx-amz-date:{}\n",
            host,
            hex::encode(Sha256::digest(payload)),
            amz_date
        );
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let payload_hash = hex::encode(Sha256::digest(payload));

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, canonical_uri, canonical_querystring, canonical_headers, signed_headers, payload_hash
        );

        // Step 2: Create string to sign
        let credential_scope = format!("{}/{}/{}/aws4_request", date_stamp, region, service);
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}",
            algorithm,
            amz_date,
            credential_scope,
            hex::encode(Sha256::digest(canonical_request.as_bytes()))
        );

        // Step 3: Calculate signature
        let mut signing_key = Self::get_signature_key(secret_key, date_stamp, region, service)?;
        signing_key.update(string_to_sign.as_bytes());
        let signature = hex::encode(signing_key.finalize().into_bytes());

        // Step 4: Create authorization header
        let authorization_header = format!(
            "{} Credential={}/{}, SignedHeaders={}, Signature={}",
            algorithm, access_key, credential_scope, signed_headers, signature
        );

        Ok(authorization_header)
    }

    fn get_signature_key(key: &str, date_stamp: &str, region_name: &str, service_name: &str) -> Result<HmacSha256> {
        let k_date = Self::hmac_sha256(format!("AWS4{}", key).as_bytes(), date_stamp.as_bytes())?;
        let k_region = Self::hmac_sha256(&k_date, region_name.as_bytes())?;
        let k_service = Self::hmac_sha256(&k_region, service_name.as_bytes())?;
        let mac = HmacSha256::new_from_slice(&k_service)
            .map_err(|e| anyhow::anyhow!("HMAC error: {}", e))?;
        Ok(mac)
    }

    fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
        let mut mac = HmacSha256::new_from_slice(key)
            .map_err(|e| anyhow::anyhow!("HMAC error: {}", e))?;
        mac.update(data);
        Ok(mac.finalize().into_bytes().to_vec())
    }

    /// Call the Amazon Polly API
    async fn call_polly_api(
        &self,
        text: &str,
        access_key: &str,
        secret_key: &str,
        config: &TtsConfig,
    ) -> Result<PollyResponse> {
        let voice = self.get_voice(config);
        let engine = self.get_engine(config);
        let output_format = match config.output_format.as_str() {
            "mp3" => "mp3",
            "wav" => "pcm",
            "ogg_vorbis" => "ogg_vorbis",
            _ => "mp3",
        };

        let url = format!("{}/v1/speech", self.api_base);

        // Build request body
        let request_body = PollyRequest {
            text: text.to_string(),
            output_format: output_format.to_string(),
            voice_id: voice,
            engine: engine.clone(),
            language_code: if config.language != "en-US" {
                Some(config.language.clone())
            } else {
                None
            },
            sample_rate: if config.sample_rate != 24000 {
                Some(config.sample_rate.to_string())
            } else {
                None
            },
            text_type: if config.enable_ssml {
                Some("ssml".to_string())
            } else {
                None
            },
            speech_mark_types: None,
        };

        let body_bytes = serde_json::to_vec(&request_body)?;

        // Sign the request
        let authorization = self.sign_request(
            access_key,
            secret_key,
            "POST",
            "polly",
            &self.region,
            &body_bytes,
        )?;

        let now = chrono::Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();

        let timeout = Duration::from_secs(config.timeout_secs);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Amz-Date", amz_date)
            .header("X-Amz-Content-Sha256", hex::encode(Sha256::digest(&body_bytes)))
            .header("Authorization", authorization)
            .header("Host", "polly.amazonaws.com")
            .body(body_bytes)
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to Amazon Polly API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Amazon Polly API request failed: {status} - {error_text}");
        }

        // Read audio data directly
        let audio_data = response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .context("Failed to read Polly response")?;

        Ok(PollyResponse {
            audio_data,
            content_type: format!("audio/{}", output_format),
        })
    }

    /// Convert response to TtsResult
    fn convert_response(response: PollyResponse, config: &TtsConfig) -> TtsResult {
        TtsResult {
            audio_data: response.audio_data,
            format: config.output_format.clone(),
            sample_rate: config.sample_rate,
            channels: 1,
            duration: None,
            timestamps: Vec::new(),
        }
    }
}

impl Default for AmazonTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for AmazonTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::Amazon
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Get credentials
        let (access_key, secret_key) = self.get_credentials(config)?;

        // Build SSML if needed
        let text_to_use = if config.enable_ssml {
            text.to_string()
        } else {
            text_to_ssml(text, config)
        };

        // Call API
        let response = self
            .call_polly_api(&text_to_use, &access_key, &secret_key, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate SSML
        if ssml.is_empty() {
            anyhow::bail!("SSML is empty");
        }

        // Get credentials
        let (access_key, secret_key) = self.get_credentials(config)?;

        // Create config with SSML enabled
        let ssml_config = TtsConfig {
            enable_ssml: true,
            ..config.clone()
        };

        // Call API
        let response = self
            .call_polly_api(ssml, &access_key, &secret_key, &ssml_config)
            .await?;

        // Convert response
        Ok(Self::convert_response(response, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("AWS_ACCESS_KEY_ID").is_ok() && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok()
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // Return a subset of popular Amazon Polly voices
        let mut voices = vec![
            VoiceInfo {
                id: "Joanna".to_string(),
                name: "Joanna".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(false),
            },
            VoiceInfo {
                id: "Matthew".to_string(),
                name: "Matthew".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(false),
            },
            VoiceInfo {
                id: "Amy".to_string(),
                name: "Amy".to_string(),
                language: "en-GB".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "Emma".to_string(),
                name: "Emma".to_string(),
                language: "en-GB".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "Lupe".to_string(),
                name: "Lupe".to_string(),
                language: "es-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(false),
            },
            VoiceInfo {
                id: "Penelope".to_string(),
                name: "Penelope".to_string(),
                language: "es-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "Vicki".to_string(),
                name: "Vicki".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "Kendra".to_string(),
                name: "Kendra".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
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
// AMAZON POLLY API TYPES
// ═══════════════════════════════════════════════════════════════

/// Polly request
#[derive(Debug, Serialize)]
struct PollyRequest {
    #[serde(rename = "Text")]
    text: String,
    #[serde(rename = "OutputFormat")]
    output_format: String,
    #[serde(rename = "VoiceId")]
    voice_id: String,
    #[serde(rename = "Engine")]
    engine: String,
    #[serde(rename = "LanguageCode")]
    language_code: Option<String>,
    #[serde(rename = "SampleRate")]
    sample_rate: Option<String>,
    #[serde(rename = "TextType")]
    text_type: Option<String>,
    #[serde(rename = "SpeechMarkTypes")]
    speech_mark_types: Option<String>,
}

/// Polly response
#[derive(Debug)]
struct PollyResponse {
    audio_data: Vec<u8>,
    content_type: String,
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amazon_provider_new() {
        let provider = AmazonTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::Amazon);
    }

    #[test]
    fn test_amazon_provider_default() {
        let provider = AmazonTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::Amazon);
    }

    #[test]
    fn test_amazon_provider_name() {
        let provider = AmazonTtsProvider::new();
        assert_eq!(provider.name(), "Amazon Polly");
    }

    #[test]
    fn test_amazon_provider_with_region() {
        let provider = AmazonTtsProvider::with_region("eu-west-1".to_string());
        assert_eq!(provider.provider_type(), TtsProviderType::Amazon);
        assert_eq!(provider.region, "eu-west-1");
    }

    #[test]
    fn test_get_voice_default() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig::default();

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "Joanna");
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig {
            voice: Some("Matthew".to_string()),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "Matthew");
    }

    #[test]
    fn test_get_engine_neural() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig {
            voice: Some("Joanna".to_string()),
            ..Default::default()
        };

        let engine = provider.get_engine(&config);
        assert_eq!(engine, "standard");
    }

    #[test]
    fn test_get_engine_neural_voice() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig {
            voice: Some("Amy".to_string()),
            ..Default::default()
        };

        let engine = provider.get_engine(&config);
        assert_eq!(engine, "neural");
    }

    #[tokio::test]
    async fn test_amazon_provider_empty_text() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_amazon_provider_too_long_text() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig::default();

        let long_text = "a".repeat(5000);
        let result = provider.synthesize(&long_text, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    #[tokio::test]
    async fn test_amazon_provider_no_credentials() {
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");

        let provider = AmazonTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("Hello world", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_amazon_provider_is_available() {
        let provider = AmazonTtsProvider::new();
        let _ = provider.is_available().await;
    }

    #[tokio::test]
    async fn test_amazon_provider_get_voices() {
        let provider = AmazonTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_amazon_provider_get_voices_with_language() {
        let provider = AmazonTtsProvider::new();
        let voices = provider.get_voices(Some("en-US")).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.language == "en-US"));
    }

    #[tokio::test]
    async fn test_amazon_provider_get_voices_no_match() {
        let provider = AmazonTtsProvider::new();
        let voices = provider.get_voices(Some("zh-CN")).await.unwrap();

        assert!(voices.is_empty());
    }

    #[test]
    fn test_convert_response_empty() {
        let response = PollyResponse {
            audio_data: vec![],
            content_type: "audio/mp3".to_string(),
        };

        let config = TtsConfig::default();
        let result = AmazonTtsProvider::convert_response(response, &config);

        assert!(result.audio_data.is_empty());
        assert_eq!(result.format, "mp3");
    }

    #[test]
    fn test_convert_response_with_data() {
        let response = PollyResponse {
            audio_data: vec![0u8, 1, 2, 3, 4, 5],
            content_type: "audio/mp3".to_string(),
        };

        let config = TtsConfig::default();
        let result = AmazonTtsProvider::convert_response(response, &config);

        assert_eq!(result.audio_data, vec![0u8, 1, 2, 3, 4, 5]);
        assert_eq!(result.format, "mp3");
    }

    #[tokio::test]
    async fn test_amazon_provider_synthesize_ssml_empty() {
        let provider = AmazonTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize_ssml("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_hmac_sha256() {
        let result = AmazonTtsProvider::hmac_sha256(b"key", b"data");
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn test_get_signature_key() {
        let result = AmazonTtsProvider::get_signature_key("secret", "20230101", "us-east-1", "polly");
        assert!(result.is_ok());
    }
}
