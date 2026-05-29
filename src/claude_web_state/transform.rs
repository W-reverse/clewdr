use std::{fmt::Write, mem};

use base64::{Engine, prelude::BASE64_STANDARD};
use futures::{StreamExt, stream};
use itertools::Itertools;
use serde_json::Value;
use tracing::warn;
use wreq::multipart::{Form, Part};

use crate::{
    claude_web_state::ClaudeWebState,
    config::CLEWDR_CONFIG,
    types::{
        claude::{
            ContentBlock, CreateMessageParams, ImageSource, Message, MessageContent, Role,
            ToolChoice,
        },
        claude_web::request::*,
    },
    utils::{TIME_ZONE, print_out_text},
};

impl ClaudeWebState {
    pub fn transform_request(&self, mut value: CreateMessageParams) -> Option<WebRequestBody> {
        let system = value.system.take();
        let msgs = mem::take(&mut value.messages);
        let system = merge_system(system.unwrap_or_default());
        let tool_choice = value.tool_choice.take();
        let user_tools = value.tools.take();
        let merged = merge_messages(msgs, system, tool_choice.as_ref(), user_tools.as_deref())?;

        let mut tools: Vec<Tool> = user_tools
            .unwrap_or_default()
            .iter()
            .filter_map(Tool::from_claude_tool)
            .collect();

        if let Some(ref tc) = tool_choice {
            match tc {
                ToolChoice::None => {
                    tools.clear();
                }
                ToolChoice::Tool { name, .. } => {
                    tools.retain(|t| t.name == *name);
                }
                _ => {}
            }
        }

        if CLEWDR_CONFIG.load().web_search {
            tools.push(Tool::web_search());
        }

        Some(WebRequestBody {
            max_tokens_to_sample: value.max_tokens,
            attachments: vec![Attachment::new(merged.paste)],
            files: vec![],
            model: if self.is_pro() {
                Some(value.model)
            } else {
                None
            },
            rendering_mode: if value.stream.unwrap_or_default() {
                "messages".to_string()
            } else {
                "raw".to_string()
            },
            prompt: merged.prompt,
            timezone: TIME_ZONE.to_string(),
            images: merged.images,
            tools,
        })
    }

    /// Upload images to the Claude.ai
    pub async fn upload_images(&self, imgs: Vec<ImageSource>) -> Vec<String> {
        // upload images
        stream::iter(imgs)
            .filter_map(async |img| {
                let ImageSource::Base64 { media_type, data } = img else {
                    warn!("Image type is not base64");
                    return None;
                };
                // decode the image
                let bytes = BASE64_STANDARD
                    .decode(data)
                    .inspect_err(|e| {
                        warn!("Failed to decode image: {}", e);
                    })
                    .ok()?;
                // choose the file name based on the media type (extract main type before any params)
                let main_type = media_type.split(';').next().unwrap_or(&media_type);
                let file_name = match main_type.to_lowercase().as_str() {
                    "image/png" => "image.png",
                    "image/jpeg" => "image.jpg",
                    "image/jpg" => "image.jpg",
                    "image/gif" => "image.gif",
                    "image/webp" => "image.webp",
                    "application/pdf" => "document.pdf",
                    _ => "file",
                };
                // create the part and form
                let part = Part::bytes(bytes).file_name(file_name);
                let form = Form::new().part("file", part);
                let endpoint = self
                    .endpoint
                    .join(&format!("api/{}/upload", self.org_uuid.as_ref()?))
                    .expect("Url parse error");
                // send the request into future
                let res = self
                    .build_request(http::Method::POST, endpoint)
                    .multipart(form)
                    .send()
                    .await
                    .inspect_err(|e| {
                        warn!("Failed to upload image: {}", e);
                    })
                    .ok()?;
                #[derive(serde::Deserialize)]
                struct UploadResponse {
                    file_uuid: String,
                }
                // get the response json
                let json = res
                    .json::<UploadResponse>()
                    .await
                    .inspect_err(|e| {
                        warn!("Failed to parse image response: {}", e);
                    })
                    .ok()?;
                // extract the file_uuid
                Some(json.file_uuid)
            })
            .collect::<Vec<_>>()
            .await
    }
}

/// Merged messages and images
#[derive(Default, Debug)]
struct Merged {
    pub paste: String,
    pub prompt: String,
    pub images: Vec<ImageSource>,
}

/// Merges multiple messages into a single text prompt, handling system instructions
/// and extracting any images from the messages
///
/// # Arguments
/// * `msgs` - Vector of messages to merge
/// * `system` - System instructions to prepend
/// * `tool_choice` - Optional tool choice configuration (used for forced tool calls)
/// * `tools` - Optional list of user-defined tools
///
/// # Returns
/// * `Option<Merged>` - Merged prompt text, images, and additional metadata, or None if merging fails
fn merge_messages(
    msgs: Vec<Message>,
    system: String,
    _tool_choice: Option<&ToolChoice>,
    _tools: Option<&[crate::types::claude::Tool]>,
) -> Option<Merged> {
    if msgs.is_empty() {
        return None;
    }
    let h = CLEWDR_CONFIG
        .load()
        .custom_h
        .to_owned()
        .unwrap_or("Human".to_string());
    let a = CLEWDR_CONFIG
        .load()
        .custom_a
        .to_owned()
        .unwrap_or("Assistant".to_string());

    let user_real_roles = CLEWDR_CONFIG.load().use_real_roles;
    let line_breaks = if user_real_roles { "\n\n\x08" } else { "\n\n" };
    let system = system.trim().to_string();
    let mut w = String::new();

    let mut imgs: Vec<ImageSource> = vec![];

    let chunks = msgs
        .into_iter()
        .filter_map(|m| match m.content {
            MessageContent::Blocks { content } => {
                // collect all text blocks, join them with new line
                let blocks = content
                    .into_iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(text.trim().to_string()),
                        ContentBlock::Image { source, .. } => {
                            match source {
                                ImageSource::Base64 { .. } => {
                                    imgs.push(source);
                                }
                                ImageSource::Url { url } => {
                                    if let Some(source) = ImageSource::from_data_url(&url) {
                                        imgs.push(source);
                                    } else {
                                        warn!("Unsupported image url source");
                                    }
                                }
                                ImageSource::File { .. } => {
                                    warn!("Image file sources are not supported");
                                }
                            }
                            None
                        }
                        ContentBlock::ImageUrl { image_url } => {
                            if let Some(source) = ImageSource::from_data_url(&image_url.url) {
                                imgs.push(source);
                            }
                            None
                        }
                        ContentBlock::ToolUse { name, input, .. } => {
                            let mut xml = format!(
                                "<antml:function_calls>\n<antml:invoke name=\"{name}\">\n"
                            );
                            if let Some(obj) = input.as_object() {
                                for (key, value) in obj {
                                    let val_str = match value {
                                        serde_json::Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    };
                                    xml.push_str(&format!(
                                        "<antml:parameter name=\"{key}\">{val_str}</antml:parameter>\n"
                                    ));
                                }
                            }
                            xml.push_str("</antml:invoke>\n</antml:function_calls>\n");
                            Some(xml)
                        }
                        ContentBlock::ServerToolUse { name, input, .. } => {
                            let mut xml = format!(
                                "<antml:function_calls>\n<antml:invoke name=\"{name}\">\n"
                            );
                            if let Some(obj) = input.as_object() {
                                for (key, value) in obj {
                                    let val_str = match value {
                                        serde_json::Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    };
                                    xml.push_str(&format!(
                                        "<antml:parameter name=\"{key}\">{val_str}</antml:parameter>\n"
                                    ));
                                }
                            }
                            xml.push_str("</antml:invoke>\n</antml:function_calls>\n");
                            Some(xml)
                        }
                        ContentBlock::McpToolUse { name, input, .. } => {
                            let mut xml = format!(
                                "<antml:function_calls>\n<antml:invoke name=\"{name}\">\n"
                            );
                            if let Some(obj) = input.as_object() {
                                for (key, value) in obj {
                                    let val_str = match value {
                                        serde_json::Value::String(s) => s.clone(),
                                        other => other.to_string(),
                                    };
                                    xml.push_str(&format!(
                                        "<antml:parameter name=\"{key}\">{val_str}</antml:parameter>\n"
                                    ));
                                }
                            }
                            xml.push_str("</antml:invoke>\n</antml:function_calls>\n");
                            Some(xml)
                        }
                        ContentBlock::ToolResult {
                            content, ..
                        } => {
                            let text_content = match content {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            Some(format!("<function_results>\n{text_content}\n</function_results>\n"))
                        }
                        ContentBlock::McpToolResult { content, .. } => {
                            let text_content = match content {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            Some(format!("<function_results>\n{text_content}\n</function_results>\n"))
                        }
                        ContentBlock::WebSearchToolResult { content, .. }
                        | ContentBlock::WebFetchToolResult { content, .. }
                        | ContentBlock::CodeExecutionToolResult { content, .. }
                        | ContentBlock::BashCodeExecutionToolResult { content, .. }
                        | ContentBlock::TextEditorCodeExecutionToolResult { content, .. }
                        | ContentBlock::ToolSearchToolResult { content, .. } => {
                            let text_content = match content {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            Some(format!("<function_results>\n{text_content}\n</function_results>\n"))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if blocks.is_empty() {
                    None
                } else {
                    Some((m.role, blocks))
                }
            }
            MessageContent::Text { content } => {
                // plain text
                let content = content.trim().to_string();
                if content.is_empty() {
                    None
                } else {
                    Some((m.role, content))
                }
            }
        })
        // chunk by role
        .chunk_by(|m| m.0);
    // join same role with new line
    let mut msgs = chunks.into_iter().map(|(role, grp)| {
        let txt = grp.into_iter().map(|m| m.1).collect::<Vec<_>>().join("\n");
        (role, txt)
    });
    // first message does not need prefix
    if !system.is_empty() {
        w += system.as_str();
    } else {
        let first = msgs.next()?;
        w += first.1.as_str();
    }
    for (role, text) in msgs {
        let prefix = match role {
            Role::System => {
                warn!("System message should be merged into the first message");
                continue;
            }
            Role::User => format!("{h}: "),
            Role::Assistant => format!("{a}: "),
        };
        write!(w, "{line_breaks}{prefix}{text}").ok()?;
    }
    print_out_text(w.to_owned(), "paste.txt");

    // prompt polyfill
    let p = CLEWDR_CONFIG.load().custom_prompt.to_owned();

    Some(Merged {
        paste: w,
        prompt: p,
        images: imgs,
    })
}

/// Merges system message content into a single string
/// Handles both string and array formats for system messages
///
/// # Arguments
/// * `sys` - System message content as a JSON Value
///
/// # Returns
/// Merged system message as a string
fn merge_system(sys: Value) -> String {
    match sys {
        Value::String(s) => s,
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v["text"].as_str())
            .map(|v| v.trim())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
