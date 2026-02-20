//! Async Cancellation (inspired by Codex `async-utils` crate).
//!
//! Provides `OrCancelExt` trait for cancelling async operations using
//! a `CancellationToken`. This is used to implement timeouts and graceful
//! shutdown for LLM calls and code execution.

use std::future::Future;

use tokio_util::sync::CancellationToken;

// ── CancelErr ─────────────────────────────────────────────────────────────

/// Error returned when a future is cancelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelErr {
    /// The cancellation token was triggered before the future completed.
    Cancelled,
    /// The operation timed out.
    Timeout(std::time::Duration),
}

impl std::fmt::Display for CancelErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "operation cancelled"),
            Self::Timeout(d) => write!(f, "operation timed out after {:.1}s", d.as_secs_f64()),
        }
    }
}

impl std::error::Error for CancelErr {}

// ── OrCancelExt ───────────────────────────────────────────────────────────

/// Extension trait for futures that can be cancelled.
///
/// ```ignore
/// use cancellation::OrCancelExt;
///
/// let token = CancellationToken::new();
/// let result = some_async_op().or_cancel(&token).await;
/// ```
pub trait OrCancelExt: Sized {
    type Output;

    /// Run this future, returning `Err(CancelErr::Cancelled)` if the token
    /// is triggered before the future completes.
    fn or_cancel(
        self,
        token: &CancellationToken,
    ) -> impl Future<Output = Result<Self::Output, CancelErr>> + Send;

    /// Run this future with a timeout, returning `Err(CancelErr::Timeout)`
    /// if the duration elapses before the future completes.
    fn with_timeout(
        self,
        duration: std::time::Duration,
    ) -> impl Future<Output = Result<Self::Output, CancelErr>> + Send;
}

impl<F> OrCancelExt for F
where
    F: Future + Send,
    F::Output: Send,
{
    type Output = F::Output;

    fn or_cancel(
        self,
        token: &CancellationToken,
    ) -> impl Future<Output = Result<Self::Output, CancelErr>> + Send {
        let token = token.clone();
        async move {
            tokio::select! {
                _ = token.cancelled() => Err(CancelErr::Cancelled),
                result = self => Ok(result),
            }
        }
    }

    fn with_timeout(
        self,
        duration: std::time::Duration,
    ) -> impl Future<Output = Result<Self::Output, CancelErr>> + Send {
        async move {
            match tokio::time::timeout(duration, self).await {
                Ok(result) => Ok(result),
                Err(_) => Err(CancelErr::Timeout(duration)),
            }
        }
    }
}

// ── Convenience Functions ─────────────────────────────────────────────────

/// Create a new cancellation token.
pub fn cancel_token() -> CancellationToken {
    CancellationToken::new()
}

/// Create a child token that is cancelled when the parent is cancelled.
pub fn child_token(parent: &CancellationToken) -> CancellationToken {
    parent.child_token()
}

/// Run a future with both a cancellation token and a timeout.
/// Returns `Err(CancelErr)` with whichever fires first.
pub async fn with_cancel_and_timeout<F, T>(
    future: F,
    token: &CancellationToken,
    timeout: std::time::Duration,
) -> Result<T, CancelErr>
where
    F: Future<Output = T> + Send,
    T: Send,
{
    let token = token.clone();
    tokio::select! {
        _ = token.cancelled() => Err(CancelErr::Cancelled),
        result = tokio::time::timeout(timeout, future) => {
            match result {
                Ok(val) => Ok(val),
                Err(_) => Err(CancelErr::Timeout(timeout)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn future_completes_before_cancel() {
        let token = cancel_token();
        let result = async { 42 }.or_cancel(&token).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn cancel_fires_before_future() {
        let token = cancel_token();
        token.cancel(); // Cancel immediately

        let result = async {
            sleep(Duration::from_secs(10)).await;
            42
        }
        .or_cancel(&token)
        .await;

        assert_eq!(result, Err(CancelErr::Cancelled));
    }

    #[tokio::test]
    async fn timeout_fires_for_slow_future() {
        let result = async {
            sleep(Duration::from_secs(10)).await;
            42
        }
        .with_timeout(Duration::from_millis(10))
        .await;

        assert!(matches!(result, Err(CancelErr::Timeout(_))));
    }

    #[tokio::test]
    async fn fast_future_beats_timeout() {
        let result = async { 42 }.with_timeout(Duration::from_secs(10)).await;

        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn child_token_cancelled_with_parent() {
        let parent = cancel_token();
        let child = child_token(&parent);

        parent.cancel();

        let result = async {
            sleep(Duration::from_secs(10)).await;
            42
        }
        .or_cancel(&child)
        .await;

        assert_eq!(result, Err(CancelErr::Cancelled));
    }

    #[tokio::test]
    async fn with_cancel_and_timeout_cancel_wins() {
        let token = cancel_token();
        token.cancel();

        let result = with_cancel_and_timeout(
            async {
                sleep(Duration::from_secs(10)).await;
                42
            },
            &token,
            Duration::from_secs(10),
        )
        .await;

        assert_eq!(result, Err(CancelErr::Cancelled));
    }

    #[tokio::test]
    async fn with_cancel_and_timeout_timeout_wins() {
        let token = cancel_token();

        let result = with_cancel_and_timeout(
            async {
                sleep(Duration::from_secs(10)).await;
                42
            },
            &token,
            Duration::from_millis(10),
        )
        .await;

        assert!(matches!(result, Err(CancelErr::Timeout(_))));
    }

    #[tokio::test]
    async fn with_cancel_and_timeout_future_wins() {
        let token = cancel_token();

        let result = with_cancel_and_timeout(async { 42 }, &token, Duration::from_secs(10)).await;

        assert_eq!(result, Ok(42));
    }
}
