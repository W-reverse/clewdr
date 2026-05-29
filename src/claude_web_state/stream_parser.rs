use crate::types::claude::{
    ContentBlock, ContentBlockDelta, MessageDeltaContent, MessageStartContent, Role, StopReason,
    StreamEvent, StreamUsage, Usage,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserPhase {
    Idle,
    Text { block_index: usize },
    ToolCall { tool_count: usize },
    Thinking { block_index: usize },
    Finished,
}

#[derive(Debug)]
struct ToolUseBuilder {
    block_index: usize,
    tool_use_id: String,
    tool_name: String,
    accumulated_json: String,
    json_buffer: String,
    params: Vec<(String, String)>,
    current_param_name: Option<String>,
    current_param_value: String,
    started: bool,
}

impl ToolUseBuilder {
    fn new(block_index: usize, tool_name: String, tool_use_id: String) -> Self {
        Self {
            block_index,
            tool_use_id,
            tool_name,
            accumulated_json: String::new(),
            json_buffer: String::new(),
            params: Vec::new(),
            current_param_name: None,
            current_param_value: String::new(),
            started: false,
        }
    }

    fn add_param(&mut self, name: String, value: String) {
        self.params.push((name, value));
    }

    fn finalize_json(&mut self) -> String {
        let mut json = String::from("{");
        for (i, (name, value)) in self.params.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('"');
            json.push_str(&escape_json_string(name));
            json.push_str("\":");
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(value);
            match parsed {
                Ok(serde_json::Value::String(s)) => {
                    json.push('"');
                    json.push_str(&escape_json_string(&s));
                    json.push('"');
                }
                Ok(other) => {
                    json.push_str(&other.to_string());
                }
                Err(_) => {
                    json.push('"');
                    json.push_str(&escape_json_string(value));
                    json.push('"');
                }
            }
        }
        json.push('}');
        self.json_buffer = json.clone();
        json
    }

    fn emit_events(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        if !self.started {
            let content_block = ContentBlock::ToolUse {
                id: self.tool_use_id.clone(),
                name: self.tool_name.clone(),
                input: serde_json::Value::Object(Default::default()),
                cache_control: None,
                caller: None,
            };
            events.push(StreamEvent::ContentBlockStart {
                index: self.block_index,
                content_block,
            });
            self.started = true;
        }

        let pending_json = self.finalize_json();
        if !pending_json.is_empty() {
            events.push(StreamEvent::ContentBlockDelta {
                index: self.block_index,
                delta: ContentBlockDelta::InputJsonDelta {
                    partial_json: pending_json,
                },
            });
            self.accumulated_json = self.json_buffer.clone();
        }

        events.push(StreamEvent::ContentBlockStop {
            index: self.block_index,
        });

        events
    }
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out
}

pub struct StreamParser {
    phase: ParserPhase,
    message_id: String,
    model: String,
    block_counter: usize,
    tool_use_id_counter: usize,
    buffer: String,
    tool_use_ids: Vec<String>,
    current_tool_use: Option<ToolUseBuilder>,
    usage: Usage,
}

impl StreamParser {
    pub fn new(model: String, usage: Usage) -> Self {
        Self {
            phase: ParserPhase::Idle,
            message_id: format!("msg_{}", uuid::Uuid::new_v4()),
            model,
            block_counter: 0,
            tool_use_id_counter: 0,
            buffer: String::new(),
            tool_use_ids: Vec::new(),
            current_tool_use: None,
            usage,
        }
    }

    pub fn message_id(&self) -> &str {
        &self.message_id
    }

    pub fn message_start(&self) -> StreamEvent {
        StreamEvent::MessageStart {
            message: MessageStartContent {
                id: self.message_id.clone(),
                type_: "message".to_string(),
                role: Role::Assistant,
                content: vec![],
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: Some(self.usage.clone()),
            },
        }
    }

    pub fn tool_use_ids(&self) -> &[String] {
        &self.tool_use_ids
    }

    fn next_tool_use_id(&mut self) -> String {
        self.tool_use_id_counter += 1;
        format!("toolu_{}", uuid::Uuid::new_v4())
    }

    pub fn feed_completion(
        &mut self,
        completion: &str,
    ) -> Vec<StreamEvent> {
        self.buffer.push_str(completion);
        let mut events = Vec::new();

        loop {
            let buf = self.buffer.clone();
            match self.phase {
                ParserPhase::Idle => {
                    let trimmed = buf.trim_start();
                    if trimmed.is_empty() {
                        self.buffer.clear();
                        break;
                    }

                    if trimmed.starts_with("<antml:thinking>") || trimmed.starts_with("<antthinking>") {
                        let block_index = self.block_counter;
                        self.block_counter += 1;
                        events.push(StreamEvent::ContentBlockStart {
                            index: block_index,
                            content_block: ContentBlock::Thinking {
                                signature: String::new(),
                                thinking: String::new(),
                            },
                        });
                        self.phase = ParserPhase::Thinking { block_index };
                        let tag_end = trimmed.find('>').map(|i| i + 1).unwrap_or(trimmed.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    if trimmed.starts_with("<antml:function_calls>") || trimmed.starts_with("<function_calls>") {
                        self.phase = ParserPhase::ToolCall { tool_count: 0 };
                        let tag_end = trimmed.find('>').map(|i| i + 1).unwrap_or(trimmed.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    let block_index = self.block_counter;
                    self.block_counter += 1;
                    events.push(StreamEvent::ContentBlockStart {
                        index: block_index,
                        content_block: ContentBlock::text(""),
                    });
                    self.phase = ParserPhase::Text { block_index };
                    self.buffer = buf.to_string();
                    continue;
                }
                ParserPhase::Text { block_index } => {
                    if let Some(tag_idx) = buf.find("<antml:") {
                        let before = &buf[..tag_idx];
                        if !before.is_empty() {
                            events.push(StreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: ContentBlockDelta::TextDelta {
                                    text: before.to_string(),
                                },
                            });
                        }
                        events.push(StreamEvent::ContentBlockStop { index: block_index });
                        self.buffer = buf[tag_idx..].to_string();
                        self.phase = ParserPhase::Idle;
                        continue;
                    }

                    if let Some(tag_idx) = buf.find("<function_calls>") {
                        let before = &buf[..tag_idx];
                        if !before.is_empty() {
                            events.push(StreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: ContentBlockDelta::TextDelta {
                                    text: before.to_string(),
                                },
                            });
                        }
                        events.push(StreamEvent::ContentBlockStop { index: block_index });
                        self.buffer = buf[tag_idx..].to_string();
                        self.phase = ParserPhase::Idle;
                        continue;
                    }

                    if let Some(tag_idx) = buf.find("<antml:thinking>") {
                        let before = &buf[..tag_idx];
                        if !before.is_empty() {
                            events.push(StreamEvent::ContentBlockDelta {
                                index: block_index,
                                delta: ContentBlockDelta::TextDelta {
                                    text: before.to_string(),
                                },
                            });
                        }
                        events.push(StreamEvent::ContentBlockStop { index: block_index });
                        self.buffer = buf[tag_idx..].to_string();
                        self.phase = ParserPhase::Idle;
                        continue;
                    }

                    if !buf.is_empty() {
                        events.push(StreamEvent::ContentBlockDelta {
                            index: block_index,
                            delta: ContentBlockDelta::TextDelta {
                                text: buf.clone(),
                            },
                        });
                        self.buffer.clear();
                    }
                    break;
                }
                ParserPhase::ToolCall { tool_count: _ } => {
                    let trimmed = buf.trim_start();
                    if trimmed.starts_with("</antml:function_calls>") || trimmed.starts_with("</function_calls>") {
                        if let Some(mut tool) = self.current_tool_use.take() {
                            events.extend(tool.emit_events());
                        }
                        self.phase = ParserPhase::Idle;
                        let tag_end = trimmed.find('>').map(|i| i + 1).unwrap_or(trimmed.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    if let Some(invoke_start) = find_tag_start(&buf, "antml:invoke") {
                        if invoke_start > 0 {
                            self.buffer = buf[invoke_start..].to_string();
                            continue;
                        }

                        if let Some(mut tool) = self.current_tool_use.take() {
                            events.extend(tool.emit_events());
                        }

                        let tool_name = extract_attr(&buf, "name");
                        let block_index = self.block_counter;
                        self.block_counter += 1;
                        let tool_use_id = self.next_tool_use_id();
                        self.tool_use_ids.push(tool_use_id.clone());
                        let builder = ToolUseBuilder::new(block_index, tool_name, tool_use_id);

                        self.current_tool_use = Some(builder);

                        let tag_end = buf.find('>').map(|i| i + 1).unwrap_or(buf.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    if let Some(param_start) = find_tag_start(&buf, "antml:parameter") {
                        let after_tag = &buf[param_start..];
                        let tag_end = after_tag.find('>').map(|i| i + 1).unwrap_or(after_tag.len());
                        let tag_content = &after_tag[..tag_end];
                        let param_name = extract_attr(tag_content, "name");

                        if let Some(ref mut tool) = self.current_tool_use {
                            if let Some(ref name) = tool.current_param_name {
                                let value = tool.current_param_value.clone();
                                tool.add_param(name.clone(), value);
                            }
                            tool.current_param_name = Some(param_name);
                            tool.current_param_value = String::new();
                        }

                        self.buffer = buf[param_start + tag_end..].to_string();
                        continue;
                    }

                    if let Some(close_tag) = find_closing_tag(&buf, "antml:parameter") {
                        if let Some(ref mut tool) = self.current_tool_use {
                            if let Some(ref name) = tool.current_param_name {
                                let value = tool.current_param_value.clone();
                                tool.add_param(name.clone(), value);
                                tool.current_param_name = None;
                                tool.current_param_value = String::new();
                            }
                        }
                        let tag_end = buf[close_tag..].find('>').map(|i| i + close_tag + 1).unwrap_or(buf.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    if let Some(close_tag) = find_closing_tag(&buf, "antml:invoke") {
                        if let Some(mut tool) = self.current_tool_use.take() {
                            events.extend(tool.emit_events());
                        }
                        let tag_end = buf[close_tag..].find('>').map(|i| i + close_tag + 1).unwrap_or(buf.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    if let Some(ref mut tool) = self.current_tool_use {
                        if tool.current_param_name.is_some() {
                            let next_tag = buf.find('<').unwrap_or(buf.len());
                            if next_tag > 0 {
                                tool.current_param_value.push_str(&buf[..next_tag]);
                                self.buffer = buf[next_tag..].to_string();
                                continue;
                            }
                        }
                    }

                    break;
                }
                ParserPhase::Thinking { block_index } => {
                    if buf.contains("</antml:thinking>") || buf.contains("</antthinking>") {
                        events.push(StreamEvent::ContentBlockStop {
                            index: block_index,
                        });
                        self.phase = ParserPhase::Idle;
                        let tag_pos = buf.find("</antml:thinking>")
                            .or_else(|| buf.find("</antthinking>"))
                            .unwrap_or(0);
                        let tag_end = buf[tag_pos..].find('>').map(|i| i + tag_pos + 1).unwrap_or(buf.len());
                        self.buffer = buf[tag_end..].to_string();
                        continue;
                    }

                    let next = buf.find('<').unwrap_or(buf.len());
                    if next > 0 {
                        let text = &buf[..next];
                        events.push(StreamEvent::ContentBlockDelta {
                            index: block_index,
                            delta: ContentBlockDelta::ThinkingDelta {
                                thinking: text.to_string(),
                            },
                        });
                        self.buffer = buf[next..].to_string();
                        continue;
                    }
                    break;
                }
                ParserPhase::Finished => break,
            }
        }

        events
    }

    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        match self.phase {
            ParserPhase::Text { block_index } => {
                events.push(StreamEvent::ContentBlockStop { index: block_index });
            }
            ParserPhase::ToolCall { .. } => {
                if let Some(mut tool) = self.current_tool_use.take() {
                    events.extend(tool.emit_events());
                }
            }
            ParserPhase::Thinking { block_index } => {
                events.push(StreamEvent::ContentBlockStop { index: block_index });
            }
            _ => {}
        }

        let stop_reason = if !self.tool_use_ids.is_empty() {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        events.push(StreamEvent::MessageDelta {
            delta: MessageDeltaContent {
                stop_reason: Some(stop_reason),
                stop_sequence: None,
            },
            usage: Some(StreamUsage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
            }),
        });
        events.push(StreamEvent::MessageStop);

        self.phase = ParserPhase::Finished;
        events
    }

    pub fn is_finished(&self) -> bool {
        self.phase == ParserPhase::Finished
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_use_ids.is_empty()
    }
}

fn find_tag_start(buf: &str, tag_name: &str) -> Option<usize> {
    let open = format!("<{}", tag_name);
    let self_close = format!("<{}/>", tag_name);
    buf.find(&open).and_then(|idx| {
        if buf[idx..].starts_with(&self_close) {
            None
        } else {
            Some(idx)
        }
    })
}

fn find_closing_tag(buf: &str, tag_name: &str) -> Option<usize> {
    let close = format!("</{}", tag_name);
    buf.find(&close)
}

fn extract_attr(tag_content: &str, attr_name: &str) -> String {
    let pattern = format!("{}=\"", attr_name);
    if let Some(start) = tag_content.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = tag_content[value_start..].find('"') {
            return tag_content[value_start..value_start + end].to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_json_string() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json_string("line\nbreak"), "line\\nbreak");
    }

    #[test]
    fn test_extract_attr() {
        let tag = "<antml:invoke name=\"get_weather\">";
        assert_eq!(extract_attr(tag, "name"), "get_weather");

        let param = "<antml:parameter name=\"location\">";
        assert_eq!(extract_attr(param, "name"), "location");
    }

    #[test]
    fn test_plain_text_stream() {
        let mut parser = StreamParser::new("claude-sonnet".to_string(), Usage::default());
        let events = parser.feed_completion("Hello world");
        assert!(!events.is_empty());
        assert!(parser.tool_use_ids().is_empty());

        let final_events = parser.finish();
        let has_stop = final_events
            .iter()
            .any(|e| matches!(e, StreamEvent::MessageStop));
        assert!(has_stop);
    }

    #[test]
    fn test_tool_call_detection() {
        let mut parser = StreamParser::new("claude-sonnet".to_string(), Usage::default());

        let text = "I'll check the weather\n\n<antml:function_calls>\n<antml:invoke name=\"get_weather\">\n<antml:parameter name=\"location\">San Francisco</antml:parameter>\n</antml:invoke>\n</antml:function_calls>";
        let events = parser.feed_completion(text);
        assert!(!parser.tool_use_ids().is_empty());

        let finish = parser.finish();
        let has_tool_delta = finish.iter().any(|e| {
            matches!(e, StreamEvent::MessageDelta {
                delta: MessageDeltaContent { stop_reason: Some(StopReason::ToolUse), .. },
                ..
            })
        });
        assert!(has_tool_delta);
    }
}
