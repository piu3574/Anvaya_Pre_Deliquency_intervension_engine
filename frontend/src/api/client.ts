import axios from "axios";
import type {
  CustomerRow,
  DashboardStatsResponse,
  ModelHealthSummary,
  ModelHealthTimelinePoint,
  PipelineRun,
  PipelineStage,
  PipelineStatus,
  RiskLogsResponse,
  RuntimeConfig
} from "../types";

const dashboardBase = import.meta.env.VITE_API_DASHBOARD_URL ?? "http://localhost:8001";
const scoringBase = import.meta.env.VITE_API_SCORING_URL ?? "http://localhost:8000";

export const dashboardApi = axios.create({
  baseURL: dashboardBase,
  timeout: 12000
});

export const scoringApi = axios.create({
  baseURL: scoringBase,
  timeout: 12000
});

export async function getDashboardStats(): Promise<DashboardStatsResponse> {
  const { data } = await dashboardApi.get<DashboardStatsResponse>("/stats");
  return data;
}

export async function getCustomers(limit = 120, offset = 0, riskBand?: string): Promise<CustomerRow[]> {
  const params: Record<string, string | number> = { limit, offset };
  if (riskBand) params.risk_band = riskBand;
  const { data } = await dashboardApi.get<CustomerRow[]>("/customers", { params });
  return data;
}

export async function getScoreByCustomer(customerId: string) {
  const { data } = await scoringApi.get(`/score/${customerId}`);
  return data;
}

export async function getRiskLogs(limit = 100, offset = 0, riskBand?: string): Promise<RiskLogsResponse> {
  const params: Record<string, string | number> = { limit, offset };
  if (riskBand) params.risk_band = riskBand;
  const { data } = await dashboardApi.get<RiskLogsResponse>("/logs/risk", { params });
  return data;
}

export async function getPipelineStatus(): Promise<PipelineStatus> {
  const { data } = await dashboardApi.get<PipelineStatus>("/pipeline/status");
  return data;
}

export async function getPipelineRuns(hours = 24): Promise<PipelineRun[]> {
  const { data } = await dashboardApi.get<{ items: PipelineRun[] }>("/pipeline/runs", { params: { hours } });
  return data.items;
}

export async function getPipelineStagesLatest(): Promise<PipelineStage[]> {
  const { data } = await dashboardApi.get<{ items: PipelineStage[] }>("/pipeline/stages/latest");
  return data.items;
}

export async function getModelHealthSummary(): Promise<ModelHealthSummary> {
  const { data } = await dashboardApi.get<ModelHealthSummary>("/model-health/summary");
  return data;
}

export async function getModelHealthTimeline(hours = 24): Promise<ModelHealthTimelinePoint[]> {
  const { data } = await dashboardApi.get<{ items: ModelHealthTimelinePoint[] }>("/model-health/timeline", {
    params: { hours }
  });
  return data.items;
}

export async function getRuntimeConfig(): Promise<RuntimeConfig> {
  const { data } = await dashboardApi.get<RuntimeConfig>("/config/runtime");
  return data;
}
