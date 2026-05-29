use serde::{Deserialize, Serialize};

use crate::types::claude::ImageSource;

/// Claude.ai attachment
#[derive(Deserialize, Serialize, Debug)]
pub struct Attachment {
    extracted_content: String,
    file_name: String,
    file_type: String,
    file_size: u64,
}

impl Attachment {
    /// Creates a new Attachment with the given content
    ///
    /// # Arguments
    /// * `content` - The text content for the attachment
    ///
    /// # Returns
    /// A new Attachment instance configured as a text file
    pub fn new(content: String) -> Self {
        Attachment {
            file_size: content.len() as u64,
            extracted_content: content,
            file_name: "paste.txt".to_string(),
            file_type: "txt".to_string(),
        }
    }
}

/// Request body to be sent to the Claude.ai
#[derive(Deserialize, Serialize, Debug)]
pub struct WebRequestBody {
    pub max_tokens_to_sample: u32,
    pub attachments: Vec<Attachment>,
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub rendering_mode: String,
    pub prompt: String,
    pub timezone: String,
    #[serde(skip)]
    pub images: Vec<ImageSource>,
    pub tools: Vec<Tool>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Tool {
    pub name: String,
    #[serde(rename = "type", skip_serializing_if = "should_skip_type")]
    pub type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

fn should_skip_type(type_: &str) -> bool {
    type_ == "custom" || type_.is_empty()
}

impl Tool {
    pub fn web_search() -> Self {
        Tool {
            name: "web_search".to_string(),
            type_: "web_search_v0".to_string(),
            description: None,
            input_schema: None,
        }
    }

    pub fn from_claude_tool(tool: &crate::types::claude::Tool) -> Option<Self> {
        use crate::types::claude::Tool as ClaudeTool;
        match tool {
            ClaudeTool::Custom(ct) => Some(Tool {
                name: ct.name.clone(),
                type_: ct
                    .type_
                    .map(|t| match t {
                        crate::types::claude::CustomToolType::Custom => "custom".to_string(),
                    })
                    .unwrap_or_else(|| "custom".to_string()),
                description: ct.description.clone(),
                input_schema: Some(ct.input_schema.clone()),
            }),
            ClaudeTool::Known(kt) => {
                let (name, type_) = match kt {
                    crate::types::claude::KnownTool::Bash20250124 { .. } => {
                        ("bash".to_string(), "bash_20250124".to_string())
                    }
                    crate::types::claude::KnownTool::TextEditor20250124 { .. }
                    | crate::types::claude::KnownTool::TextEditor20250429 { .. }
                    | crate::types::claude::KnownTool::TextEditor20250728 { .. } => {
                        ("str_replace_editor".to_string(), "text_editor_20250124".to_string())
                    }
                    crate::types::claude::KnownTool::WebSearch20250305 { .. } => {
                        ("web_search".to_string(), "web_search_20250305".to_string())
                    }
                };
                Some(Tool {
                    name,
                    type_,
                    description: None,
                    input_schema: Some(serde_json::json!({"type": "object", "properties": {}})),
                })
            }
            ClaudeTool::Raw(v) => {
                let name = v
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let type_ = v
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("custom")
                    .to_string();
                let description = v
                    .get("description")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string());
                let input_schema = v.get("input_schema").cloned();
                if name.is_empty() {
                    None
                } else {
                    Some(Tool {
                        type_,
                        name,
                        description,
                        input_schema,
                    })
                }
            }
        }
    }
}
