use async_stream::try_stream;
use axum::{
    BoxError, Json,
    response::{IntoResponse, Sse, sse::Event as SseEvent},
};
use bytes::Bytes;
use eventsource_stream::{EventStream, Eventsource};
use futures::{Stream, StreamExt, TryStreamExt};
use serde::Deserialize;
use snafu::ResultExt;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::debug;
use url::Url;
use wreq::{Method, Proxy};

use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::{
        ClaudeWebState,
        session::SessionManager,
        sse_manager::RawEventStream,
        stream_parser::StreamParser,
    },
    config::CLEWDR_CONFIG,
    error::{CheckClaudeErr, ClewdrError, WreqSnafu},
    types::claude::{
        ContentBlock, CountMessageTokensResponse, CreateMessageParams, CreateMessageResponse,
        Message, MessageStartContent, Role, StopReason, StreamEvent, Usage,
    },
    utils::print_out_text,
};

pub async fn merge_sse(
    stream: EventStream<impl Stream<Item = Result<Bytes, wreq::Error>>>,
) -> Result<String, ClewdrError> {
    #[derive(Deserialize)]
    struct Data {
        completion: String,
    }
    Ok(stream
        .try_filter_map(async |event| {
            Ok(serde_json::from_str::<Data>(&event.data)
                .map(|data| data.completion)
                .ok())
        })
        .try_collect()
        .await?)
}

impl<S> From<S> for Message
where
    S: Into<String>,
{
    fn from(str: S) -> Self {
        Message::new_blocks(Role::Assistant, vec![ContentBlock::text(str.into())])
    }
}

fn is_valid_stream_content_block(block: &ContentBlock) -> bool {
    !matches!(
        block,
        ContentBlock::ToolResult { .. }
            | ContentBlock::ServerToolUse { .. }
            | ContentBlock::WebSearchToolResult { .. }
            | ContentBlock::WebFetchToolResult { .. }
            | ContentBlock::CodeExecutionToolResult { .. }
            | ContentBlock::BashCodeExecutionToolResult { .. }
            | ContentBlock::TextEditorCodeExecutionToolResult { .. }
            | ContentBlock::ToolSearchToolResult { .. }
            | ContentBlock::McpToolUse { .. }
            | ContentBlock::McpToolResult { .. }
            | ContentBlock::ContainerUpload { .. }
    )
}

fn filter_event(event: &StreamEvent) -> bool {
    match event {
        StreamEvent::ContentBlockStart { content_block, .. } => {
            is_valid_stream_content_block(content_block)
        }
        StreamEvent::Error { .. } => false,
        _ => true,
    }
}

fn stream_event_to_sse(event: StreamEvent) -> Result<SseEvent, serde_json::Error> {
    let (event_type, data) = match &event {
        StreamEvent::MessageStart { .. } => ("message_start", serde_json::to_string(&event)?),
        StreamEvent::ContentBlockStart { .. } => ("content_block_start", serde_json::to_string(&event)?),
        StreamEvent::ContentBlockDelta { .. } => ("content_block_delta", serde_json::to_string(&event)?),
        StreamEvent::ContentBlockStop { .. } => ("content_block_stop", serde_json::to_string(&event)?),
        StreamEvent::MessageDelta { .. } => ("message_delta", serde_json::to_string(&event)?),
        StreamEvent::MessageStop => ("message_stop", serde_json::to_string(&event)?),
        StreamEvent::Ping => ("ping", serde_json::to_string(&event)?),
        StreamEvent::Error { .. } => ("error", serde_json::to_string(&event)?),
    };
    Ok(SseEvent::default().event(event_type).data(data))
}

#[derive(Deserialize)]
struct CompletionData {
    completion: String,
}

fn spawn_stream_parser_task(
    mut raw_stream: RawEventStream,
    model: String,
    usage: Usage,
    event_tx: mpsc::UnboundedSender<StreamEvent>,
) {
    tokio::spawn(async move {
        let mut parser = StreamParser::new(model, usage);
        let _ = event_tx.send(parser.message_start());

        while let Some(event_result) = raw_stream.next().await {
            match event_result {
                Ok(event) => {
                    tracing::debug!(
                        "SSE event: type={}, data={}",
                        event.event,
                        &event.data[..event.data.len().min(200)]
                    );
                    if let Ok(data) = serde_json::from_str::<CompletionData>(&event.data) {
                        if data.completion.is_empty() {
                            continue;
                        }
                        let events = parser.feed_completion(&data.completion);
                        for e in events {
                            if event_tx.send(e).is_err() {
                                return;
                            }
                        }
                    } else {
                        tracing::debug!("SSE data not JSON completion: {}", &event.data[..event.data.len().min(200)]);
                    }
                }
                Err(e) => {
                    tracing::error!("SSE stream error: {}", e);
                    let _ = event_tx.send(StreamEvent::Error {
                        error: crate::types::claude::StreamError {
                            type_: "internal_error".to_string(),
                            message: e.to_string(),
                        },
                    });
                    return;
                }
            }
        }

        let final_events = parser.finish();
        for event in final_events {
            if event_tx.send(event).is_err() {
                return;
            }
        }
    });
}

impl ClaudeWebState {
    pub async fn transform_response_with_opts(
        &mut self,
        wreq_res: wreq::Response,
        has_tools: bool,
    ) -> Result<axum::response::Response, ClewdrError> {
        if self.stream {
            if has_tools {
                return self.transform_stream_with_tools(wreq_res).await;
            }
            return self.transform_stream_simple(wreq_res).await;
        }

        let stream = wreq_res.bytes_stream();
        let stream = stream.eventsource();
        let text = merge_sse(stream).await?;
        print_out_text(text.to_owned(), "claude_web_non_stream.txt");
        let mut response =
            CreateMessageResponse::text(text.clone(), Default::default(), self.usage.to_owned());

        let enable_precise = CLEWDR_CONFIG.load().enable_web_count_tokens;
        let mut usage = self.usage.to_owned();
        if enable_precise && let Some(inp) = self.try_code_count_tokens().await {
            usage.input_tokens = inp;
        }
        let mut output_tokens = response.count_tokens();
        if enable_precise && let Some(model) = self.last_params.as_ref().map(|p| p.model.clone()) {
            let out = count_code_output_tokens_for_text(
                self.cookie.clone(),
                self.endpoint.clone(),
                self.proxy.clone(),
                self.client.clone(),
                model,
                text.clone(),
                self.cookie_actor_handle.clone(),
            )
            .await;
            if let Some(v) = out {
                output_tokens = v;
            }
        }
        usage.output_tokens = output_tokens;
        response.usage = Some(usage.clone());
        self.persist_usage_totals(usage.input_tokens as u64, output_tokens as u64)
            .await;
        Ok(Json(response).into_response())
    }

    async fn transform_stream_simple(
        &mut self,
        wreq_res: wreq::Response,
    ) -> Result<axum::response::Response, ClewdrError> {
        let raw_stream = crate::claude_web_state::sse_manager::wrap_response_stream(wreq_res);

        let stream = try_stream! {
            let mut sse_stream = raw_stream;
            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(event) => {
                        if let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) {
                            if filter_event(&parsed) {
                                yield stream_event_to_sse(parsed)
                                    .map_err(|e| axum::Error::new(e))?;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("SSE stream error in simple: {}", e);
                        break;
                    }
                }
            }
        };

        let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
        Ok(Sse::new(stream)
            .keep_alive(Default::default())
            .into_response())
    }

    async fn transform_stream_with_tools(
        &mut self,
        wreq_res: wreq::Response,
    ) -> Result<axum::response::Response, ClewdrError> {
        let raw_stream = crate::claude_web_state::sse_manager::wrap_response_stream(wreq_res);
        let (event_tx, event_rx) = mpsc::unbounded_channel::<StreamEvent>();
        let shared_rx: crate::claude_web_state::session::SharedEventReceiver =
            Arc::new(Mutex::new(event_rx));

        tokio::spawn(async move {
            let tx = event_tx;
            let mut sse_stream = raw_stream;
            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(event) => {
                        if let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) {
                            if filter_event(&parsed) {
                                if tx.send(parsed).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("SSE stream error in tools: {}", e);
                        break;
                    }
                }
            }
        });

        let model = self
            .last_params
            .as_ref()
            .map(|p| p.model.clone())
            .unwrap_or_default();

        let message_start = MessageStartContent {
            id: format!("msg_{}", uuid::Uuid::new_v4()),
            type_: "message".to_string(),
            role: Role::Assistant,
            content: vec![],
            model,
            stop_reason: None,
            stop_sequence: None,
            usage: Some(self.usage.clone()),
        };

        let mut tool_use_ids = Vec::new();
        let mut events_to_stream = Vec::new();
        let mut reached_tool_use = false;
        let mut last_block_was_tool_use = false;
        let mut active_tool_use_block_index: Option<usize> = None;
        let tool_use_timeout = std::time::Duration::from_millis(1000);

        loop {
            let event = tokio::time::timeout(tool_use_timeout, async {
                let mut rx = shared_rx.lock().await;
                rx.recv().await
            })
            .await;

            match event {
                Ok(Some(ref event)) => {
                    if let StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlock::ToolUse { id, .. },
                        ..
                    } = event
                    {
                        tool_use_ids.push(id.clone());
                        active_tool_use_block_index = Some(*index);
                        last_block_was_tool_use = false;
                    }

                    if let StreamEvent::ContentBlockStop { index } = event {
                        if active_tool_use_block_index == Some(*index) {
                            last_block_was_tool_use = true;
                            active_tool_use_block_index = None;
                        }
                    }

                    if let StreamEvent::ContentBlockStart { .. } = event {
                        last_block_was_tool_use = false;
                    }

                    let is_tool_use_delta = matches!(
                        event,
                        StreamEvent::MessageDelta {
                            delta: crate::types::claude::MessageDeltaContent {
                                stop_reason: Some(StopReason::ToolUse),
                                ..
                            },
                            ..
                        }
                    );

                    if is_tool_use_delta {
                        reached_tool_use = true;
                    }

                    let is_stop = matches!(event, StreamEvent::MessageStop);
                    events_to_stream.push(event.clone());

                    if reached_tool_use {
                        let mut rx = shared_rx.lock().await;
                        while let Ok(event) = rx.try_recv() {
                            events_to_stream.push(event);
                        }
                        break;
                    }

                    if is_stop {
                        break;
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(_timeout) => {
                    if last_block_was_tool_use {
                        reached_tool_use = true;
                        let mut rx = shared_rx.lock().await;
                        while let Ok(event) = rx.try_recv() {
                            events_to_stream.push(event);
                        }
                        break;
                    }
                    last_block_was_tool_use = false;
                }
            }
        }

        if reached_tool_use && !tool_use_ids.is_empty() {
            SessionManager::pause(
                self.clone(),
                message_start,
                shared_rx,
                tool_use_ids,
            )
            .await;
        }

        let events = events_to_stream;
        let stream = try_stream! {
            for event in events {
                yield stream_event_to_sse(event).map_err(|e| axum::Error::new(e))?;
            }
        };
        let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
        Ok(Sse::new(stream)
            .keep_alive(Default::default())
            .into_response())
    }

    pub async fn resume_after_tool_result(
        &mut self,
        tool_use_id: &str,
        tool_result_content: &str,
    ) -> Result<axum::response::Response, ClewdrError> {
        let (resumed_state, message_start, _shared_rx) =
            SessionManager::resume_by_tool_id(tool_use_id)
                .await
                .ok_or(ClewdrError::BadRequest {
                    msg: "No pending tool call found for this tool_use_id",
                })?;

        *self = resumed_state;

        let org_uuid = self
            .org_uuid
            .to_owned()
            .ok_or(ClewdrError::UnexpectedNone {
                msg: "Organization UUID is not set",
            })?;

        let conv_uuid = self
            .conv_uuid
            .to_owned()
            .ok_or(ClewdrError::UnexpectedNone {
                msg: "Conversation UUID is not set",
            })?;

        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}/tool_result",
                org_uuid, conv_uuid
            ))
            .map_err(|e| ClewdrError::Whatever {
                message: format!("Parse URL error: {e}"),
                source: Some(Box::new(e)),
            })?;

        let body = serde_json::json!({
            "tool_use_id": tool_use_id,
            "content": [{"type": "text", "text": tool_result_content}],
        });

        let resp = self
            .build_request(Method::POST, endpoint)
            .json(&body)
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to send tool result",
            })?;

        if !resp.status().is_success() {
            return resp.check_claude().await.map(|_| {
                axum::response::Response::new(axum::body::Body::empty())
            });
        }

        debug!("Tool result sent, continuing with completion");

        let raw_stream = crate::claude_web_state::sse_manager::wrap_response_stream(resp);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<StreamEvent>();

        tokio::spawn(async move {
            let tx = event_tx;
            let mut sse_stream = raw_stream;
            while let Some(event_result) = sse_stream.next().await {
                match event_result {
                    Ok(event) => {
                        if let Ok(parsed) = serde_json::from_str::<StreamEvent>(&event.data) {
                            if filter_event(&parsed) {
                                if tx.send(parsed).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("SSE stream error in resume: {}", e);
                        break;
                    }
                }
            }
        });

        let stream = try_stream! {
            yield stream_event_to_sse(StreamEvent::MessageStart {
                message: message_start,
            }).map_err(|e| axum::Error::new(e))?;

            loop {
                match event_rx.recv().await {
                    Some(event) => {
                        let is_done = matches!(&event, StreamEvent::MessageStop);
                        yield stream_event_to_sse(event).map_err(|e| axum::Error::new(e))?;
                        if is_done {
                            break;
                        }
                    }
                    None => break,
                }
            }
        };

        if let Some(conv_uuid) = self.conv_uuid.clone() {
            SessionManager::complete(&conv_uuid).await;
        }

        let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
        Ok(Sse::new(stream)
            .keep_alive(Default::default())
            .into_response())
    }
}

async fn bearer_count_tokens(
    state: &ClaudeCodeState,
    access_token: &str,
    body: &CreateMessageParams,
) -> Option<u32> {
    let url = state.endpoint.join("v1/messages/count_tokens").ok()?;
    let resp = state
        .client
        .post(url.to_string())
        .bearer_auth(access_token)
        .header("anthropic-version", "2023-06-01")
        .json(body)
        .send()
        .await
        .ok()?;
    let resp = resp.check_claude().await.ok()?;
    let v: CountMessageTokensResponse = resp.json().await.ok()?;
    Some(v.input_tokens)
}

impl ClaudeWebState {
    pub(crate) async fn try_code_count_tokens(&mut self) -> Option<u32> {
        self.cookie.as_ref()?;
        let params = self.last_params.as_ref()?.clone();
        let mut code = ClaudeCodeState::new(self.cookie_actor_handle.clone());
        code.cookie = self.cookie.clone();
        code.endpoint = self.endpoint.clone();
        code.proxy = self.proxy.clone();
        code.client = self.client.clone();
        if let Some(ref c) = self.cookie
            && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
        {
            code.set_cookie_header_value(val);
        }

        let org = code.get_organization().await.ok()?;
        let exch = code.exchange_code(&org).await.ok()?;
        code.exchange_token(exch).await.ok()?;
        let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

        let mut body = params.clone();
        body.stream = Some(false);

        bearer_count_tokens(&code, &access, &body).await
    }
}

async fn count_code_output_tokens_for_text(
    cookie: Option<crate::config::CookieStatus>,
    endpoint: Url,
    proxy: Option<Proxy>,
    client: wreq::Client,
    model: String,
    text: String,
    handle: crate::services::cookie_actor::CookieActorHandle,
) -> Option<u32> {
    let mut code = ClaudeCodeState::new(handle.clone());
    code.cookie = cookie.clone();
    code.endpoint = endpoint;
    code.proxy = proxy;
    code.client = client;
    if let Some(ref c) = cookie
        && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
    {
        code.set_cookie_header_value(val);
    }
    let org = code.get_organization().await.ok()?;
    let exch = code.exchange_code(&org).await.ok()?;
    code.exchange_token(exch).await.ok()?;
    let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

    let body = CreateMessageParams {
        model,
        messages: vec![Message::new_text(Role::Assistant, text)],
        ..Default::default()
    };
    bearer_count_tokens(&code, &access, &body).await
}
