// ═══════════════════════════════════════════════════════════════
// AZURE SPEECH SERVICES TEXT-TO-SPEECH PROVIDER
// ═══════════════════════════════════════════════════════════════

use super::{TtsConfig, TtsProvider, TtsProviderType, TtsResult, VoiceInfo, VoiceGender, validate_text};
use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

/// Azure Speech Services TTS provider
pub struct AzureTtsProvider {
    client: Client,
    api_base: String,
}

impl AzureTtsProvider {
    /// Create a new Azure TTS provider
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://*.cognitiveservices.azure.com".to_string(),
        }
    }

    /// Create a new provider with custom region
    pub fn with_region(region: String) -> Self {
        let api_base = format!("https://{region}.tts.speech.microsoft.com/cognitiveservices/v1");
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
    fn get_api_key(&self, config: &TtsConfig) -> Result<String> {
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

    /// Get the voice to use
    fn get_voice(&self, config: &TtsConfig) -> String {
        config
            .voice
            .clone()
            .unwrap_or_else(|| {
                // Default to neural voice for the language
                format!("{}-{}-Neural",
                    config.language.split('-').next().unwrap_or("en"),
                    config.language.split('-').nth(1).unwrap_or("US").to_uppercase()
                )
            })
    }

    /// Get the API URL
    fn get_api_url(&self) -> Result<String> {
        if self.api_base.contains('*') {
            let region = self.get_region()?;
            Ok(format!("https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"))
        } else {
            Ok(self.api_base.clone())
        }
    }

    /// Build SSML for Azure TTS
    fn build_ssml(&self, text: &str, voice: &str, config: &TtsConfig) -> String {
        if config.enable_ssml {
            // Assume text is already SSML, just update voice
            text.replace("voice name=", &format!("voice name=\"{}\"", voice))
        } else {
            format!(
                r#"<speak version='1.0' xml:lang='{}'><voice name='{}'><prosody rate='{}' pitch='{}'>{}</prosody></voice></speak>"#,
                config.language,
                voice,
                (config.rate * 100.0) as i32,
                config.pitch as i32,
                self.escape_xml(text)
            )
        }
    }

    /// Escape XML special characters
    fn escape_xml(&self, text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    /// Call the Azure TTS API
    async fn call_azure_api(
        &self,
        ssml: &str,
        api_key: &str,
        config: &TtsConfig,
    ) -> Result<Vec<u8>> {
        let url = self.get_api_url()?;
        let timeout = Duration::from_secs(config.timeout_secs);

        // Build output format
        let output_format = match config.output_format.as_str() {
            "mp3" => "audio-24khz-48kbitrate-mono-mp3",
            "wav" => "riff-24khz-16bit-mono-pcm",
            "opus" => "audio-24khz-48kbitrate-mono-opus",
            _ => "audio-24khz-48kbitrate-mono-mp3",
        };

        let response = self
            .client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", api_key)
            .header("Content-Type", "application/ssml+xml")
            .header("X-Microsoft-OutputFormat", output_format)
            .body(ssml.to_string())
            .timeout(timeout)
            .send()
            .await
            .context("Failed to send request to Azure TTS API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Azure TTS API request failed: {status} - {error_text}");
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .context("Failed to read Azure TTS response")
    }

    /// Convert audio data to TtsResult
    fn convert_response(audio_data: Vec<u8>, config: &TtsConfig) -> TtsResult {
        TtsResult {
            audio_data,
            format: config.output_format.clone(),
            sample_rate: config.sample_rate,
            channels: 1,
            duration: None, // Azure doesn't provide duration in response
            timestamps: Vec::new(),
        }
    }
}

impl Default for AzureTtsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TtsProvider for AzureTtsProvider {
    fn provider_type(&self) -> TtsProviderType {
        TtsProviderType::Azure
    }

    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate text
        validate_text(text)?;

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get voice
        let voice = self.get_voice(config);

        // Build SSML
        let ssml = self.build_ssml(text, &voice, config);

        // Call API
        let audio_data = self
            .call_azure_api(&ssml, &api_key, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(audio_data, config))
    }

    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult> {
        // Validate SSML
        if ssml.is_empty() {
            anyhow::bail!("SSML is empty");
        }

        // Get API key
        let api_key = self.get_api_key(config)?;

        // Get voice for SSML (in case it needs to be updated)
        let voice = self.get_voice(config);

        // Update SSML with voice if needed
        let ssml = self.build_ssml(ssml, &voice, config);

        // Call API
        let audio_data = self
            .call_azure_api(&ssml, &api_key, config)
            .await?;

        // Convert response
        Ok(Self::convert_response(audio_data, config))
    }

    async fn is_available(&self) -> bool {
        std::env::var("AZURE_SPEECH_KEY").is_ok()
    }

    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>> {
        // Return a subset of popular Azure TTS voices
        let mut voices = vec![
            VoiceInfo {
                id: "en-US-JennyNeural".to_string(),
                name: "Jenny Neural".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en-US-GuyNeural".to_string(),
                name: "Guy Neural".to_string(),
                language: "en-US".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en-GB-SoniaNeural".to_string(),
                name: "Sonia Neural".to_string(),
                language: "en-GB".to_string(),
                gender: VoiceGender::Female,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "en-GB-RyanNeural".to_string(),
                name: "Ryan Neural".to_string(),
                language: "en-GB".to_string(),
                gender: VoiceGender::Male,
                sample_rate: Some(24000),
                neural: Some(true),
            },
            VoiceInfo {
                id: "es-ES-ElviraNeural".to_string(),
                name: "Elvira Neural".to_string(),
                language: "es-ES".to_string(),
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
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_provider_new() {
        let provider = AzureTtsProvider::new();
        assert_eq!(provider.provider_type(), TtsProviderType::Azure);
    }

    #[test]
    fn test_azure_provider_default() {
        let provider = AzureTtsProvider::default();
        assert_eq!(provider.provider_type(), TtsProviderType::Azure);
    }

    #[test]
    fn test_azure_provider_name() {
        let provider = AzureTtsProvider::new();
        assert_eq!(provider.name(), "Azure Speech Services");
    }

    #[test]
    fn test_azure_provider_with_region() {
        let provider = AzureTtsProvider::with_region("westus".to_string());
        assert_eq!(provider.provider_type(), TtsProviderType::Azure);
        assert!(provider.api_base.contains("westus"));
    }

    #[test]
    fn test_get_voice_default() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig::default();

        let voice = provider.get_voice(&config);
        assert!(voice.contains("Neural"));
        assert!(voice.contains("en-US"));
    }

    #[test]
    fn test_get_voice_custom() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig {
            voice: Some("en-US-JennyNeural".to_string()),
            ..Default::default()
        };

        let voice = provider.get_voice(&config);
        assert_eq!(voice, "en-US-JennyNeural");
    }

    #[test]
    fn test_escape_xml() {
        let provider = AzureTtsProvider::new();
        let text = "Hello <world> & 'friends'";
        let escaped = provider.escape_xml(text);
        assert_eq!(escaped, "Hello &lt;world&gt; &amp; &apos;friends&apos;");
    }

    #[test]
    fn test_build_ssml_plain() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig {
            enable_ssml: false,
            ..Default::default()
        };

        let ssml = provider.build_ssml("Hello world", "en-US-JennyNeural", &config);
        assert!(ssml.contains("<speak"));
        assert!(ssml.contains("Hello world"));
        assert!(ssml.contains("en-US-JennyNeural"));
        assert!(ssml.contains("</speak>"));
    }

    #[test]
    fn test_build_ssml_already_ssml() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig {
            enable_ssml: true,
            ..Default::default()
        };

        let ssml = provider.build_ssml("<speak>Hello</speak>", "en-US-JennyNeural", &config);
        assert!(ssml.contains("en-US-JennyNeural"));
        assert!(ssml.contains("<speak"));
    }

    #[tokio::test]
    async fn test_azure_provider_empty_text() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_azure_provider_too_long_text() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig::default();

        let long_text = "a".repeat(5000);
        let result = provider.synthesize(&long_text, &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    #[tokio::test]
    async fn test_azure_provider_no_api_key() {
        std::env::remove_var("AZURE_SPEECH_KEY");

        let provider = AzureTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize("Hello world", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_azure_provider_is_available() {
        let provider = AzureTtsProvider::new();
        let _ = provider.is_available().await;
    }

    #[tokio::test]
    async fn test_azure_provider_get_voices() {
        let provider = AzureTtsProvider::new();
        let voices = provider.get_voices(None).await.unwrap();

        assert!(!voices.is_empty());
    }

    #[tokio::test]
    async fn test_azure_provider_get_voices_with_language() {
        let provider = AzureTtsProvider::new();
        let voices = provider.get_voices(Some("en-US")).await.unwrap();

        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.language == "en-US"));
    }

    #[tokio::test]
    async fn test_azure_provider_get_voices_no_match() {
        let provider = AzureTtsProvider::new();
        let voices = provider.get_voices(Some("zh-CN")).await.unwrap();

        assert!(voices.is_empty());
    }

    #[test]
    fn test_convert_response_empty() {
        let audio_data = vec![];
        let config = TtsConfig::default();
        let result = AzureTtsProvider::convert_response(audio_data, &config);

        assert!(result.audio_data.is_empty());
        assert_eq!(result.format, "mp3");
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_convert_response_with_data() {
        let audio_data = vec![0u8, 1, 2, 3, 4, 5];
        let config = TtsConfig::default();
        let result = AzureTtsProvider::convert_response(audio_data, &config);

        assert_eq!(result.audio_data, vec![0u8, 1, 2, 3, 4, 5]);
        assert_eq!(result.format, "mp3");
        assert_eq!(result.sample_rate, 24000);
    }

    #[tokio::test]
    async fn test_azure_provider_synthesize_ssml_empty() {
        let provider = AzureTtsProvider::new();
        let config = TtsConfig::default();

        let result = provider.synthesize_ssml("", &config).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_azure_provider_no_region_wildcard() {
        let provider = AzureTtsProvider::new();
        assert!(provider.api_base.contains('*'));
    }
}
