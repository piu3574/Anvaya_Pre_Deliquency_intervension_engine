export type RiskBand = "GREEN" | "YELLOW" | "RED";

export interface DashboardStatsResponse {
  total_count_in_db: number;
  sample_count: number;
  pd_final_stats: {
    min: number;
    mean: number;
    max: number;
  };
  distribution_sample: Record<string, number>;
}

export interface CustomerRow {
  customer_id: string;
  pd_final: number;
  risk_band: RiskBand;
  y_true?: number;
  created_at?: string;
}

export interface ChartBin {
  bucket: string;
  value: number;
}

export interface QueueItem {
  id: string;
  score: number;
  band: RiskBand;
  elapsed: string;
}

export interface RiskLogItem {
  id: string;
  customer_id: string;
  pd_xgb: number;
  pd_lgbm: number;
  pd_ensemble: number;
  risk_band: RiskBand;
  scored_at: string;
  source: string;
  priority: "critical" | "watch" | "normal";
}

export interface RiskLogsResponse {
  count: number;
  items: RiskLogItem[];
}

export interface PipelineStatus {
  state: "active" | "idle" | "stale" | "cold" | string;
  service: {
    status: string;
    service: string;
    version: string;
  };
  last_scored_at: string | null;
  seconds_since_last_score: number | null;
  events_last_15m: number;
  red_events_last_15m: number;
  latest_event: RiskLogItem | null;
}

export interface PipelineRun {
  run_window: string;
  total_events: number;
  critical_events: number;
  watch_events: number;
  status: "critical" | "healthy" | string;
}

export interface PipelineStage {
  stage: number;
  name: string;
  status: "ok" | "blocked" | string;
  detail: string;
}

export interface ModelHealthSummary {
  has_data: boolean;
  auc: number | null;
  mean_pd: number;
  observed_default_rate: number | null;
  calibration_error: number | null;
  drift_proxy: number;
  red_rate: number;
  yellow_rate: number;
  green_rate: number;
  events_last_hour: number;
  sample_size: number;
}

export interface ModelHealthTimelinePoint {
  time: string;
  events: number;
  avg_pd: number;
  red_events: number;
}

export interface RuntimeConfig {
  risk_thresholds: {
    green_lt: number;
    red_gte: number;
  };
  service: {
    version: string;
    dashboard_table: string;
    risk_log_table: string;
  };
}
