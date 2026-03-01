//! Observability layer: metrics collection, JSONL logging, and HTML dashboard generation.
//!
//! Provides structured logging of per-epoch training metrics and generates
//! a self-contained HTML file with interactive Chart.js charts for debugging
//! and analysis.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::time::Instant;

/// Metrics recorded at the end of each epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub mrr: Option<f64>,
    pub hits_at_1: Option<f64>,
    pub hits_at_3: Option<f64>,
    pub hits_at_10: Option<f64>,
    pub mean_rank: Option<f64>,
    pub learning_rate: f64,
    pub epoch_duration_ms: u128,
    pub num_batches: usize,
    pub avg_batch_loss: f64,
}

/// Collects training metrics across epochs and produces outputs.
pub struct MetricsLogger {
    /// All recorded epoch metrics.
    pub history: Vec<EpochMetrics>,
    /// Path for JSONL output.
    jsonl_path: Option<String>,
    /// Timer for current epoch.
    epoch_start: Option<Instant>,
    /// Batch losses accumulated within current epoch.
    batch_losses: Vec<f64>,
}

impl MetricsLogger {
    /// Create a new metrics logger.
    ///
    /// If `jsonl_path` is given, metrics are appended to that file each epoch.
    pub fn new(jsonl_path: Option<String>) -> Self {
        Self {
            history: Vec::new(),
            jsonl_path,
            epoch_start: None,
            batch_losses: Vec::new(),
        }
    }

    /// Mark the start of an epoch for timing.
    pub fn start_epoch(&mut self) {
        self.epoch_start = Some(Instant::now());
        self.batch_losses.clear();
    }

    /// Record a single batch loss within the current epoch.
    pub fn record_batch_loss(&mut self, loss: f64) {
        self.batch_losses.push(loss);
    }

    /// Finalize the epoch and record all metrics.
    pub fn end_epoch(
        &mut self,
        epoch: usize,
        lr: f64,
        eval_metrics: Option<&crate::eval::metrics::LinkPredictionMetrics>,
    ) {
        let duration = self
            .epoch_start
            .map(|s| s.elapsed().as_millis())
            .unwrap_or(0);

        let num_batches = self.batch_losses.len();
        let train_loss: f64 = if num_batches > 0 {
            self.batch_losses.iter().sum::<f64>() / num_batches as f64
        } else {
            0.0
        };
        let avg_batch_loss = train_loss;

        let metrics = EpochMetrics {
            epoch,
            train_loss,
            val_loss: eval_metrics.map(|_| train_loss), // placeholder if needed
            mrr: eval_metrics.map(|m| m.mrr),
            hits_at_1: eval_metrics.map(|m| m.hits_at_1),
            hits_at_3: eval_metrics.map(|m| m.hits_at_3),
            hits_at_10: eval_metrics.map(|m| m.hits_at_10),
            mean_rank: eval_metrics.map(|m| m.mean_rank),
            learning_rate: lr,
            epoch_duration_ms: duration,
            num_batches,
            avg_batch_loss,
        };

        // Console output
        print!(
            "  Epoch {:>3} | loss: {:.6} | lr: {:.6} | batches: {} | {:.1}s",
            metrics.epoch,
            metrics.train_loss,
            metrics.learning_rate,
            metrics.num_batches,
            metrics.epoch_duration_ms as f64 / 1000.0
        );
        if let Some(em) = eval_metrics {
            print!(
                " | MRR: {:.4} | H@1: {:.4} | H@3: {:.4} | H@10: {:.4} | MR: {:.1}",
                em.mrr, em.hits_at_1, em.hits_at_3, em.hits_at_10, em.mean_rank
            );
        }
        println!();

        // JSONL output
        if let Some(ref path) = self.jsonl_path {
            if let Ok(json) = serde_json::to_string(&metrics) {
                if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(file, "{}", json);
                }
            }
        }

        self.history.push(metrics);
    }

    /// Generate a self-contained HTML dashboard with embedded Chart.js charts.
    pub fn generate_dashboard(&self, output_path: &str) -> std::io::Result<()> {
        let metrics_json =
            serde_json::to_string(&self.history).unwrap_or_else(|_| "[]".to_string());

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HEHRGNN Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #0f0f23;
    color: #e0e0e0;
    padding: 24px;
  }}
  h1 {{
    text-align: center;
    font-size: 2rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .subtitle {{
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 32px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: #1a1a2e;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2a2a4a;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 1.1rem;
    color: #a0a0d0;
    margin-bottom: 16px;
  }}
  canvas {{ width: 100% !important; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #2a2a4a;
  }}
  th {{ color: #667eea; font-weight: 600; }}
  td {{ color: #ccc; }}
  .best {{ color: #4ade80; font-weight: bold; }}
  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }}
  .stat {{
    text-align: center;
    padding: 16px;
    background: #16213e;
    border-radius: 8px;
  }}
  .stat-value {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .stat-label {{
    font-size: 0.75rem;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
</style>
</head>
<body>
<h1>HEHRGNN Training Dashboard</h1>
<p class="subtitle">Generated at training completion</p>

<div class="grid">
  <!-- Summary Stats -->
  <div class="card full" id="summary-card">
    <h2>Training Summary</h2>
    <div class="stat-grid" id="stat-grid"></div>
  </div>

  <!-- Loss Chart -->
  <div class="card">
    <h2>Training Loss</h2>
    <canvas id="lossChart"></canvas>
  </div>

  <!-- Link Prediction Metrics -->
  <div class="card">
    <h2>Link Prediction Metrics</h2>
    <canvas id="metricsChart"></canvas>
  </div>

  <!-- Epoch Duration -->
  <div class="card">
    <h2>Epoch Duration (seconds)</h2>
    <canvas id="durationChart"></canvas>
  </div>

  <!-- Mean Rank -->
  <div class="card">
    <h2>Mean Rank</h2>
    <canvas id="rankChart"></canvas>
  </div>

  <!-- Epoch Details Table -->
  <div class="card full">
    <h2>Epoch Details</h2>
    <div style="overflow-x: auto;">
      <table id="epochTable">
        <thead>
          <tr>
            <th>Epoch</th><th>Loss</th><th>LR</th>
            <th>MRR</th><th>H@1</th><th>H@3</th><th>H@10</th>
            <th>Mean Rank</th><th>Duration</th><th>Batches</th>
          </tr>
        </thead>
        <tbody id="epochBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const data = {metrics_json};

// ---- Summary Stats ----
const statGrid = document.getElementById('stat-grid');
if (data.length > 0) {{
  const last = data[data.length - 1];
  const bestMrr = data.filter(d => d.mrr != null).reduce((a, b) => (b.mrr > a.mrr ? b : a), {{mrr: 0, epoch: 0}});
  const totalTime = data.reduce((s, d) => s + d.epoch_duration_ms, 0);
  const stats = [
    ['Total Epochs', data.length],
    ['Final Loss', last.train_loss.toFixed(6)],
    ['Best MRR', bestMrr.mrr != null ? bestMrr.mrr.toFixed(4) + ' (E' + bestMrr.epoch + ')' : 'N/A'],
    ['Final H@10', last.hits_at_10 != null ? last.hits_at_10.toFixed(4) : 'N/A'],
    ['Total Time', (totalTime / 1000).toFixed(1) + 's'],
    ['Avg Epoch', (totalTime / data.length / 1000).toFixed(2) + 's'],
  ];
  stats.forEach(([label, value]) => {{
    statGrid.innerHTML += `<div class="stat"><div class="stat-value">${{value}}</div><div class="stat-label">${{label}}</div></div>`;
  }});
}}

// ---- Chart Helpers ----
const epochs = data.map(d => d.epoch);
const chartOpts = (title) => ({{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }}, title: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#1f1f3a' }} }},
    y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#1f1f3a' }} }}
  }}
}});

// ---- Loss Chart ----
new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: epochs,
    datasets: [{{
      label: 'Train Loss',
      data: data.map(d => d.train_loss),
      borderColor: '#f87171',
      backgroundColor: 'rgba(248,113,113,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: chartOpts('Loss')
}});

// ---- Metrics Chart ----
const metricsData = data.filter(d => d.mrr != null);
if (metricsData.length > 0) {{
  new Chart(document.getElementById('metricsChart'), {{
    type: 'line',
    data: {{
      labels: metricsData.map(d => d.epoch),
      datasets: [
        {{ label: 'MRR', data: metricsData.map(d => d.mrr), borderColor: '#4ade80', tension: 0.3, pointRadius: 2 }},
        {{ label: 'Hits@1', data: metricsData.map(d => d.hits_at_1), borderColor: '#22d3ee', tension: 0.3, pointRadius: 2 }},
        {{ label: 'Hits@3', data: metricsData.map(d => d.hits_at_3), borderColor: '#a78bfa', tension: 0.3, pointRadius: 2 }},
        {{ label: 'Hits@10', data: metricsData.map(d => d.hits_at_10), borderColor: '#fbbf24', tension: 0.3, pointRadius: 2 }},
      ]
    }},
    options: chartOpts('Link Prediction')
  }});
}}

// ---- Duration Chart ----
new Chart(document.getElementById('durationChart'), {{
  type: 'bar',
  data: {{
    labels: epochs,
    datasets: [{{
      label: 'Seconds',
      data: data.map(d => d.epoch_duration_ms / 1000),
      backgroundColor: 'rgba(102,126,234,0.6)',
      borderColor: '#667eea',
      borderWidth: 1,
    }}]
  }},
  options: chartOpts('Duration')
}});

// ---- Mean Rank Chart ----
const rankData = data.filter(d => d.mean_rank != null);
if (rankData.length > 0) {{
  new Chart(document.getElementById('rankChart'), {{
    type: 'line',
    data: {{
      labels: rankData.map(d => d.epoch),
      datasets: [{{
        label: 'Mean Rank',
        data: rankData.map(d => d.mean_rank),
        borderColor: '#fb923c',
        backgroundColor: 'rgba(251,146,60,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 2,
      }}]
    }},
    options: chartOpts('Mean Rank')
  }});
}}

// ---- Epoch Table ----
const tbody = document.getElementById('epochBody');
data.forEach(d => {{
  const row = `<tr>
    <td>${{d.epoch}}</td>
    <td>${{d.train_loss.toFixed(6)}}</td>
    <td>${{d.learning_rate.toFixed(6)}}</td>
    <td>${{d.mrr != null ? d.mrr.toFixed(4) : '-'}}</td>
    <td>${{d.hits_at_1 != null ? d.hits_at_1.toFixed(4) : '-'}}</td>
    <td>${{d.hits_at_3 != null ? d.hits_at_3.toFixed(4) : '-'}}</td>
    <td>${{d.hits_at_10 != null ? d.hits_at_10.toFixed(4) : '-'}}</td>
    <td>${{d.mean_rank != null ? d.mean_rank.toFixed(1) : '-'}}</td>
    <td>${{(d.epoch_duration_ms / 1000).toFixed(2)}}s</td>
    <td>${{d.num_batches}}</td>
  </tr>`;
  tbody.innerHTML += row;
}});
</script>
</body>
</html>"#
        );

        fs::write(output_path, html)?;
        println!("\n  Dashboard written to: {}", output_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_metrics_logger_records_epochs() {
        let mut logger = MetricsLogger::new(None);

        logger.start_epoch();
        logger.record_batch_loss(0.5);
        logger.record_batch_loss(0.3);
        logger.end_epoch(0, 0.001, None);

        assert_eq!(logger.history.len(), 1);
        let m = &logger.history[0];
        assert_eq!(m.epoch, 0);
        assert!((m.train_loss - 0.4).abs() < 1e-6); // mean of 0.5, 0.3
        assert_eq!(m.num_batches, 2);
    }

    #[test]
    fn test_dashboard_generation() {
        let mut logger = MetricsLogger::new(None);
        logger.start_epoch();
        logger.record_batch_loss(1.0);
        logger.end_epoch(0, 0.001, None);

        let path = "/tmp/hehrgnn_test_dashboard.html";
        logger.generate_dashboard(path).unwrap();
        assert!(Path::new(path).exists());

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("chart.js") || content.contains("Chart"));
        assert!(content.contains("HEHRGNN Training Dashboard"));

        // Cleanup
        let _ = fs::remove_file(path);
    }
}
