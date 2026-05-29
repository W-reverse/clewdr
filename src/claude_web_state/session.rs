use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, RwLock, mpsc};

use crate::claude_web_state::ClaudeWebState;
use crate::types::claude::{MessageStartContent, StreamEvent};

pub type SharedEventReceiver = Arc<Mutex<mpsc::UnboundedReceiver<StreamEvent>>>;

pub struct PausedSession {
    pub state: ClaudeWebState,
    pub message_start: MessageStartContent,
    pub event_rx: Option<SharedEventReceiver>,
    pub tool_use_ids: Vec<String>,
    pub created_at: Instant,
}

static SESSION_STORE: LazyLock<RwLock<HashMap<String, PausedSession>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static TOOL_INDEX: LazyLock<RwLock<HashMap<String, String>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

const SESSION_TIMEOUT: Duration = Duration::from_secs(300);
const CLEANUP_INTERVAL: Duration = Duration::from_secs(60);

pub struct SessionManager;

impl SessionManager {
    pub async fn pause(
        state: ClaudeWebState,
        message_start: MessageStartContent,
        event_rx: SharedEventReceiver,
        tool_use_ids: Vec<String>,
    ) -> String {
        let conv_uuid = state
            .conv_uuid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        tracing::info!(
            "SessionManager::pause conv_uuid={}, tool_use_ids={:?}",
            conv_uuid,
            tool_use_ids
        );

        let session = PausedSession {
            state,
            message_start,
            event_rx: Some(event_rx),
            tool_use_ids: tool_use_ids.clone(),
            created_at: Instant::now(),
        };

        for tool_id in &tool_use_ids {
            TOOL_INDEX
                .write()
                .await
                .insert(tool_id.clone(), conv_uuid.clone());
        }

        SESSION_STORE.write().await.insert(conv_uuid.clone(), session);

        conv_uuid
    }

    pub async fn resume_by_tool_id(
        tool_use_id: &str,
    ) -> Option<(ClaudeWebState, MessageStartContent, SharedEventReceiver)> {
        let conv_uuid = TOOL_INDEX.read().await.get(tool_use_id).cloned();
        tracing::info!(
            "SessionManager::resume_by_tool_id tool_use_id={}, conv_uuid={:?}",
            tool_use_id,
            conv_uuid
        );

        let conv_uuid = conv_uuid?;

        let mut store = SESSION_STORE.write().await;
        let session = store.get_mut(&conv_uuid);
        if session.is_none() {
            tracing::warn!("Session not found in store for conv_uuid={}", conv_uuid);
            return None;
        }
        let session = session.unwrap();

        let rx = session.event_rx.take();
        if rx.is_none() {
            tracing::warn!("event_rx already taken for conv_uuid={}", conv_uuid);
            return None;
        }

        tracing::info!("SessionManager::resume_by_tool_id SUCCESS conv_uuid={}", conv_uuid);
        Some((session.state.clone(), session.message_start.clone(), rx.unwrap()))
    }

    pub async fn complete(conv_uuid: &str) {
        let mut store = SESSION_STORE.write().await;
        if let Some(session) = store.remove(conv_uuid) {
            let mut index = TOOL_INDEX.write().await;
            for tool_id in &session.tool_use_ids {
                index.remove(tool_id);
            }
        }
    }

    pub async fn cleanup_expired() {
        let now = Instant::now();
        let mut store = SESSION_STORE.write().await;
        let expired: Vec<String> = store
            .iter()
            .filter(|(_, s)| now.duration_since(s.created_at) > SESSION_TIMEOUT)
            .map(|(k, _)| k.clone())
            .collect();

        for key in &expired {
            if let Some(session) = store.remove(key) {
                let mut index = TOOL_INDEX.write().await;
                for tool_id in &session.tool_use_ids {
                    index.remove(tool_id);
                }
            }
        }
    }

    pub async fn cleanup_loop() {
        loop {
            tokio::time::sleep(CLEANUP_INTERVAL).await;
            Self::cleanup_expired().await;
        }
    }
}
