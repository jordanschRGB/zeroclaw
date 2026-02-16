// ═══════════════════════════════════════════════════════════════
// TTS MODULE - Text-to-Speech with multiple providers
// ═══════════════════════════════════════════════════════════════

mod openai;
mod elevenlabs;
mod google;
mod azure;
mod amazon;

pub use openai::OpenAiTtsProvider;
pub use elevenlabs::ElevenLabsTtsProvider;
pub use google::GoogleTtsProvider;
pub use azure::AzureTtsProvider;
pub use amazon::AmazonTtsProvider;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// TTS CONFIGURATION
// ═══════════════════════════════════════════════════════════════

/// TTS provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsProviderType {
    OpenAi,
    ElevenLabs,
    Google,
    Azure,
    Amazon,
}

/// Voice gender
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

/// TTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Provider type
    pub provider: TtsProviderType,
    /// API key (for cloud providers)
    pub api_key: Option<String>,
    /// Voice ID or name
    pub voice: Option<String>,
    /// Language code (e.g., "en-US", "es-ES")
    pub language: String,
    /// Model name (provider-specific)
    pub model: Option<String>,
    /// Speaking rate (0.1 to 2.0, where 1.0 is normal)
    pub rate: f32,
    /// Pitch adjustment (-20.0 to 20.0 semitones)
    pub pitch: f32,
    /// Volume boost (0 to 100 dB)
    pub volume_gain_db: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Output format (mp3, wav, opus, pcm)
    pub output_format: String,
    /// Enable SSML markup
    pub enable_ssml: bool,
    /// Timeout in seconds
    pub timeout_secs: u64,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            provider: TtsProviderType::OpenAi,
            api_key: None,
            voice: None,
            language: "en-US".to_string(),
            model: None,
            rate: 1.0,
            pitch: 0.0,
            volume_gain_db: 0.0,
            sample_rate: 24000,
            output_format: "mp3".to_string(),
            enable_ssml: false,
            timeout_secs: 30,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TTS RESULTS
// ═══════════════════════════════════════════════════════════════

/// Text-to-speech result
#[derive(Debug, Clone)]
pub struct TtsResult {
    /// Generated audio data
    pub audio_data: Vec<u8>,
    /// Audio format (mp3, wav, etc.)
    pub format: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Duration in seconds
    pub duration: Option<f32>,
    /// Word timestamps (if available)
    pub timestamps: Vec<TtsTimestamp>,
}

/// Word timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsTimestamp {
    /// Word or character
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

// ═══════════════════════════════════════════════════════════════
// TTS PROVIDER TRAIT
// ═══════════════════════════════════════════════════════════════

/// Text-to-Speech provider trait
#[async_trait]
pub trait TtsProvider: Send + Sync {
    /// Get the provider type
    fn provider_type(&self) -> TtsProviderType;

    /// Synthesize speech from text
    async fn synthesize(&self, text: &str, config: &TtsConfig) -> Result<TtsResult>;

    /// Synthesize speech with SSML
    async fn synthesize_ssml(&self, ssml: &str, config: &TtsConfig) -> Result<TtsResult>;

    /// Check if the provider is available
    async fn is_available(&self) -> bool;

    /// Get available voices
    async fn get_voices(&self, language: Option<&str>) -> Result<Vec<VoiceInfo>>;

    /// Get the provider name
    fn name(&self) -> &str {
        match self.provider_type() {
            TtsProviderType::OpenAi => "OpenAI TTS",
            TtsProviderType::ElevenLabs => "ElevenLabs",
            TtsProviderType::Google => "Google Cloud Text-to-Speech",
            TtsProviderType::Azure => "Azure Speech Services",
            TtsProviderType::Amazon => "Amazon Polly",
        }
    }
}

/// Voice information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    /// Voice ID
    pub id: String,
    /// Voice name
    pub name: String,
    /// Language code
    pub language: String,
    /// Voice gender
    pub gender: VoiceGender,
    /// Sample rate
    pub sample_rate: Option<u32>,
    /// Neural/standard voice
    pub neural: Option<bool>,
}

// ═══════════════════════════════════════════════════════════════
// TTS ENGINE
// ═══════════════════════════════════════════════════════════════

/// Main TTS engine that manages multiple providers
pub struct TtsEngine {
    providers: Vec<Box<dyn TtsProvider>>,
    default_config: TtsConfig,
}

impl TtsEngine {
    /// Create a new TTS engine with all available providers
    pub fn new(config: TtsConfig) -> Self {
        let mut providers: Vec<Box<dyn TtsProvider>> = Vec::new();

        // Add OpenAI provider
        providers.push(Box::new(OpenAiTtsProvider::new()));

        // Add ElevenLabs provider
        providers.push(Box::new(ElevenLabsTtsProvider::new()));

        // Add Google provider
        providers.push(Box::new(GoogleTtsProvider::new()));

        // Add Azure provider
        providers.push(Box::new(AzureTtsProvider::new()));

        // Add Amazon provider
        providers.push(Box::new(AmazonTtsProvider::new()));

        Self {
            providers,
            default_config: config,
        }
    }

    /// Create an TTS engine with only the default provider
    pub fn with_default_provider(config: TtsConfig) -> Self {
        let provider: Box<dyn TtsProvider> = match config.provider {
            TtsProviderType::OpenAi => Box::new(OpenAiTtsProvider::new()),
            TtsProviderType::ElevenLabs => Box::new(ElevenLabsTtsProvider::new()),
            TtsProviderType::Google => Box::new(GoogleTtsProvider::new()),
            TtsProviderType::Azure => Box::new(AzureTtsProvider::new()),
            TtsProviderType::Amazon => Box::new(AmazonTtsProvider::new()),
        };

        Self {
            providers: vec![provider],
            default_config: config,
        }
    }

    /// Get the default provider
    pub fn default_provider(&self) -> Option<&dyn TtsProvider> {
        self.providers.first().map(|p| p.as_ref())
    }

    /// Get all available providers
    pub fn available_providers(&self) -> Vec<&dyn TtsProvider> {
        self.providers.iter().map(|p| p.as_ref()).collect()
    }

    /// Synthesize speech using the default provider
    pub async fn synthesize(&self, text: &str) -> Result<TtsResult> {
        self.synthesize_with_config(text, &self.default_config).await
    }

    /// Synthesize speech with custom configuration
    pub async fn synthesize_with_config(
        &self,
        text: &str,
        config: &TtsConfig,
    ) -> Result<TtsResult> {
        // Find the matching provider
        let provider = self
            .providers
            .iter()
            .find(|p| p.provider_type() == config.provider)
            .context("Provider not found")?;

        provider.synthesize(text, config).await
    }

    /// Synthesize speech with automatic fallback
    pub async fn synthesize_with_fallback(
        &self,
        text: &str,
        config: &TtsConfig,
    ) -> Result<TtsResult> {
        let mut last_error = None;

        // Try the primary provider first
        if let Some(provider) = self
            .providers
            .iter()
            .find(|p| p.provider_type() == config.provider)
        {
            match provider.synthesize(text, config).await {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }

        // Try fallback providers
        for provider in &self.providers {
            if provider.provider_type() == config.provider {
                continue; // Skip the already-tried provider
            }

            match provider.synthesize(text, config).await {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }

        // All providers failed
        match last_error {
            Some(error) => Err(error.context("All TTS providers failed")),
            None => Err(anyhow::anyhow!("All TTS providers failed")),
        }
    }

    /// Get the default configuration
    pub fn config(&self) -> &TtsConfig {
        &self.default_config
    }
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::new(TtsConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

/// Convert text to SSML format
pub fn text_to_ssml(text: &str, config: &TtsConfig) -> String {
    if config.enable_ssml {
        // Assume text is already SSML
        text.to_string()
    } else {
        // Wrap in speak tag
        format!(
            "<speak><prosody rate=\"{}\" pitch=\"{}\">{}</prosody></speak>",
            (config.rate * 100.0) as i32,
            config.pitch,
            escape_xml(text)
        )
    }
}

/// Escape XML special characters
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Validate text input
pub fn validate_text(text: &str) -> Result<()> {
    if text.is_empty() {
        anyhow::bail!("Text is empty");
    }

    if text.len() > 4096 {
        anyhow::bail!("Text is too long (max 4096 characters)");
    }

    // Check for invalid characters (control characters except newline, tab, carriage return)
    for (i, ch) in text.char_indices() {
        if ch < ' ' && ch != '\n' && ch != '\t' && ch != '\r' {
            anyhow::bail!("Text contains invalid control character at position {}", i);
        }
    }

    Ok(())
}

/// Calculate approximate audio duration from text
pub fn estimate_duration(text: &str, rate: f32) -> f32 {
    // Average speaking rate is about 150 words per minute
    let word_count = text.split_whitespace().count();
    let words_per_second = 150.0 / 60.0;
    let base_duration = word_count as f32 / words_per_second;
    base_duration / rate
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_config_default() {
        let config = TtsConfig::default();
        assert_eq!(config.provider, TtsProviderType::OpenAi);
        assert_eq!(config.language, "en-US");
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.rate, 1.0);
        assert_eq!(config.pitch, 0.0);
        assert!(!config.enable_ssml);
    }

    #[test]
    fn test_tts_config_serialization() {
        let config = TtsConfig {
            provider: TtsProviderType::Google,
            api_key: Some("test-key".to_string()),
            voice: Some("en-US-Neural2".to_string()),
            language: "en-GB".to_string(),
            model: Some("en-GB-Neural2".to_string()),
            rate: 1.2,
            pitch: 5.0,
            volume_gain_db: 3.0,
            sample_rate: 22050,
            output_format: "wav".to_string(),
            enable_ssml: true,
            timeout_secs: 60,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TtsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.provider, TtsProviderType::Google);
        assert_eq!(parsed.api_key, Some("test-key".to_string()));
        assert_eq!(parsed.voice, Some("en-US-Neural2".to_string()));
        assert_eq!(parsed.language, "en-GB");
        assert_eq!(parsed.rate, 1.2);
        assert_eq!(parsed.pitch, 5.0);
        assert!(parsed.enable_ssml);
        assert_eq!(parsed.sample_rate, 22050);
        assert_eq!(parsed.output_format, "wav");
    }

    #[test]
    fn test_tts_timestamp_empty() {
        let timestamp = TtsTimestamp {
            text: String::new(),
            start_time: 0.0,
            end_time: 0.0,
        };

        assert!(timestamp.text.is_empty());
        assert_eq!(timestamp.start_time, 0.0);
        assert_eq!(timestamp.end_time, 0.0);
    }

    #[test]
    fn test_voice_info_serialization() {
        let voice = VoiceInfo {
            id: "voice-123".to_string(),
            name: "Test Voice".to_string(),
            language: "en-US".to_string(),
            gender: VoiceGender::Female,
            sample_rate: Some(24000),
            neural: Some(true),
        };

        let json = serde_json::to_string(&voice).unwrap();
        let parsed: VoiceInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "voice-123");
        assert_eq!(parsed.name, "Test Voice");
        assert_eq!(parsed.language, "en-US");
        assert_eq!(parsed.gender, VoiceGender::Female);
        assert_eq!(parsed.sample_rate, Some(24000));
        assert_eq!(parsed.neural, Some(true));
    }

    #[test]
    fn test_voice_gender_equality() {
        assert_eq!(VoiceGender::Male, VoiceGender::Male);
        assert_ne!(VoiceGender::Male, VoiceGender::Female);
    }

    #[test]
    fn test_voice_gender_serialization() {
        let male = VoiceGender::Male;
        let json = serde_json::to_string(&male).unwrap();
        let parsed: VoiceGender = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, VoiceGender::Male);
    }

    #[test]
    fn test_tts_provider_type_equality() {
        assert_eq!(TtsProviderType::OpenAi, TtsProviderType::OpenAi);
        assert_ne!(TtsProviderType::OpenAi, TtsProviderType::Google);
    }

    #[test]
    fn test_tts_provider_type_serialization() {
        let openai = TtsProviderType::OpenAi;
        let json = serde_json::to_string(&openai).unwrap();
        let parsed: TtsProviderType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, TtsProviderType::OpenAi);
    }

    #[test]
    fn test_escape_xml() {
        let text = "Hello <world> & 'friends'";
        let escaped = escape_xml(text);
        assert_eq!(escaped, "Hello &lt;world&gt; &amp; &apos;friends&apos;");
    }

    #[test]
    fn test_text_to_ssml_plain() {
        let config = TtsConfig {
            enable_ssml: false,
            ..Default::default()
        };

        let ssml = text_to_ssml("Hello world", &config);
        assert!(ssml.contains("<speak>"));
        assert!(ssml.contains("Hello world"));
        assert!(ssml.contains("</speak>"));
        assert!(ssml.contains("rate=\"100\""));
        assert!(ssml.contains("pitch=\"0\""));
    }

    #[test]
    fn test_text_to_ssml_already_ssml() {
        let config = TtsConfig {
            enable_ssml: true,
            ..Default::default()
        };

        let ssml = text_to_ssml("<speak>Hello</speak>", &config);
        assert_eq!(ssml, "<speak>Hello</speak>");
    }

    #[test]
    fn test_validate_text_empty() {
        let result = validate_text("");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_validate_text_too_long() {
        let long_text = "a".repeat(5000);
        let result = validate_text(&long_text);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too long"));
    }

    #[test]
    fn test_validate_text_valid() {
        let text = "Hello, world!";
        assert!(validate_text(text).is_ok());
    }

    #[test]
    fn test_validate_text_with_newlines() {
        let text = "Line 1\nLine 2\tTabbed";
        assert!(validate_text(text).is_ok());
    }

    #[test]
    fn test_validate_text_with_control_char() {
        let text = "Hello\x00World";
        let result = validate_text(text);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid control character"));
    }

    #[test]
    fn test_estimate_duration() {
        // "Hello world" is 2 words
        let duration = estimate_duration("Hello world", 1.0);
        // At 150 wpm, 2 words should take about 0.8 seconds
        assert!(duration > 0.5 && duration < 1.5);
    }

    #[test]
    fn test_estimate_duration_with_rate() {
        let duration_normal = estimate_duration("Hello world", 1.0);
        let duration_fast = estimate_duration("Hello world", 2.0);
        assert!(duration_fast < duration_normal);
    }

    #[test]
    fn test_estimate_duration_empty() {
        let duration = estimate_duration("", 1.0);
        assert_eq!(duration, 0.0);
    }

    #[test]
    fn test_tts_engine_default() {
        let engine = TtsEngine::default();
        assert_eq!(engine.config().provider, TtsProviderType::OpenAi);
        assert!(engine.default_provider().is_some());
        assert!(!engine.available_providers().is_empty());
    }

    #[test]
    fn test_tts_engine_with_config() {
        let config = TtsConfig {
            provider: TtsProviderType::Google,
            ..Default::default()
        };

        let engine = TtsEngine::with_default_provider(config);
        assert_eq!(engine.config().provider, TtsProviderType::Google);
    }

    #[test]
    fn test_tts_result_empty() {
        let result = TtsResult {
            audio_data: vec![],
            format: "mp3".to_string(),
            sample_rate: 24000,
            channels: 1,
            duration: None,
            timestamps: vec![],
        };

        assert!(result.audio_data.is_empty());
        assert_eq!(result.format, "mp3");
        assert_eq!(result.sample_rate, 24000);
        assert_eq!(result.channels, 1);
        assert!(result.duration.is_none());
        assert!(result.timestamps.is_empty());
    }
}
