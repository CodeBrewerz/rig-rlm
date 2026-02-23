//! Multimodal attachments for agent tasks.
//!
//! Supports images, audio/voice, and documents (PDFs, text) attached to agent tasks.
//! Attachments are passed through the Restate handler and injected into
//! the conversation history as rig `UserContent::Image` / `UserContent::Audio` / `UserContent::Document`.

use serde::{Deserialize, Serialize};

/// A file attachment sent with an agent task.
///
/// The client sends base64-encoded data with a MIME type.
/// The agent runtime converts this to the appropriate rig `UserContent` variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    /// Base64-encoded file content.
    pub data: String,
    /// MIME type, e.g. `"image/png"`, `"application/pdf"`.
    pub media_type: String,
    /// Optional filename for display/logging.
    #[serde(default)]
    pub filename: Option<String>,
}

impl Attachment {
    /// Create a new image attachment from base64 data.
    pub fn image(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            media_type: media_type.into(),
            filename: None,
        }
    }

    /// Create with a filename.
    pub fn with_filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }

    /// True if this is an image attachment.
    pub fn is_image(&self) -> bool {
        self.media_type.starts_with("image/")
    }

    /// True if this is a PDF document.
    pub fn is_pdf(&self) -> bool {
        self.media_type == "application/pdf"
    }

    /// True if this is a text-based document that can be loaded directly.
    pub fn is_text(&self) -> bool {
        matches!(
            self.media_type.as_str(),
            "text/plain" | "text/csv" | "text/html" | "text/markdown"
        )
    }

    /// True if this is an audio/voice attachment.
    pub fn is_audio(&self) -> bool {
        self.media_type.starts_with("audio/")
    }

    /// Map MIME type to rig's `ImageMediaType`, if applicable.
    pub fn image_media_type(&self) -> Option<rig::message::ImageMediaType> {
        use rig::message::ImageMediaType;
        match self.media_type.as_str() {
            "image/png" => Some(ImageMediaType::PNG),
            "image/jpeg" | "image/jpg" => Some(ImageMediaType::JPEG),
            "image/gif" => Some(ImageMediaType::GIF),
            "image/webp" => Some(ImageMediaType::WEBP),
            "image/svg+xml" => Some(ImageMediaType::SVG),
            _ => None,
        }
    }

    /// Map MIME type to rig's `AudioMediaType`, if applicable.
    pub fn audio_media_type(&self) -> Option<rig::message::AudioMediaType> {
        use rig::message::AudioMediaType;
        match self.media_type.as_str() {
            "audio/wav" | "audio/x-wav" | "audio/wave" => Some(AudioMediaType::WAV),
            "audio/mpeg" | "audio/mp3" => Some(AudioMediaType::MP3),
            "audio/aiff" | "audio/x-aiff" => Some(AudioMediaType::AIFF),
            "audio/aac" => Some(AudioMediaType::AAC),
            "audio/ogg" => Some(AudioMediaType::OGG),
            "audio/flac" => Some(AudioMediaType::FLAC),
            _ => None,
        }
    }

    /// Decode the base64 data into raw bytes.
    pub fn decode_bytes(&self) -> Result<Vec<u8>, String> {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(&self.data)
            .map_err(|e| format!("base64 decode error: {e}"))
    }

    /// Extract text content from the attachment.
    ///
    /// - **PDF**: Uses `pdf-extract` to extract text from all pages.
    /// - **Text**: Decodes base64 directly to UTF-8 text.
    /// - **Other**: Returns an error.
    pub fn extract_text(&self) -> Result<String, String> {
        if self.is_pdf() {
            let bytes = self.decode_bytes()?;
            pdf_extract::extract_text_from_mem(&bytes)
                .map(|text| text.trim().to_string())
                .map_err(|e| format!("PDF extraction error: {e}"))
        } else if self.is_text() {
            let bytes = self.decode_bytes()?;
            String::from_utf8(bytes).map_err(|e| format!("UTF-8 decode error: {e}"))
        } else {
            Err(format!(
                "Cannot extract text from {} attachment",
                self.media_type
            ))
        }
    }

    /// Display label for logging.
    pub fn label(&self) -> String {
        if let Some(ref name) = self.filename {
            format!("{} ({})", name, self.media_type)
        } else {
            format!("attachment ({})", self.media_type)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attachment_is_image() {
        let att = Attachment::image("base64data", "image/png");
        assert!(att.is_image());
        assert!(!att.is_pdf());
        assert!(!att.is_text());
        assert!(!att.is_audio());
    }

    #[test]
    fn attachment_is_audio() {
        let att = Attachment::image("base64data", "audio/wav");
        assert!(att.is_audio());
        assert!(!att.is_image());
        assert!(att.audio_media_type().is_some());

        let mp3 = Attachment::image("x", "audio/mp3");
        assert!(mp3.is_audio());
        assert!(mp3.audio_media_type().is_some());
    }

    #[test]
    fn attachment_is_pdf() {
        let att = Attachment {
            data: "base64data".to_string(),
            media_type: "application/pdf".to_string(),
            filename: Some("doc.pdf".to_string()),
        };
        assert!(!att.is_image());
        assert!(att.is_pdf());
    }

    #[test]
    fn attachment_media_type_mapping() {
        let png = Attachment::image("x", "image/png");
        assert!(png.image_media_type().is_some());

        let pdf = Attachment::image("x", "application/pdf");
        assert!(pdf.image_media_type().is_none());
    }

    #[test]
    fn attachment_serde_roundtrip() {
        let att = Attachment::image("SGVsbG8=", "image/png").with_filename("test.png");
        let json = serde_json::to_string(&att).unwrap();
        let back: Attachment = serde_json::from_str(&json).unwrap();
        assert_eq!(back.data, "SGVsbG8=");
        assert_eq!(back.media_type, "image/png");
        assert_eq!(back.filename.as_deref(), Some("test.png"));
    }

    #[test]
    fn text_attachment_extraction() {
        use base64::Engine;
        let text = "Hello, world!\nLine 2.";
        let encoded = base64::engine::general_purpose::STANDARD.encode(text);
        let att = Attachment {
            data: encoded,
            media_type: "text/plain".to_string(),
            filename: Some("hello.txt".to_string()),
        };
        assert!(att.is_text());
        let extracted = att.extract_text().unwrap();
        assert_eq!(extracted, text);
    }

    #[test]
    fn image_extract_text_fails_gracefully() {
        let att = Attachment::image("base64data", "image/png");
        let result = att.extract_text();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("image/png"));
    }
}
