import type { ComponentType } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Bell,
  LayoutDashboard,
  ListChecks,
  SlidersHorizontal,
  Search,
  Settings,
  Shield,
  UserCircle2,
  Users,
  Waves
} from "lucide-react";
import DashboardPage from "./pages/DashboardPage";
import ExplainabilityPage from "./pages/ExplainabilityPage";
import ModelHealthPage from "./pages/ModelHealthPage";
import PartyRiskListPage from "./pages/PartyRiskListPage";
import PipelineMonitorPage from "./pages/PipelineMonitorPage";
import ConfigurationPage from "./pages/ConfigurationPage";

type NavItem = {
  label: string;
  path: string;
  icon: ComponentType<{ size?: string | number }>;
};

const navItems: NavItem[] = [
  { label: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
  { label: "Pipeline Monitor", path: "/pipeline-monitor", icon: Activity },
  { label: "Party Risk List", path: "/party-risk-list", icon: Users },
  { label: "Explainability", path: "/explainability", icon: Waves },
  { label: "Model Health", path: "/model-health", icon: BarChart3 },
  { label: "Live Risk Logs", path: "/live-risk-logs", icon: ListChecks },
  { label: "Configuration", path: "/configuration", icon: SlidersHorizontal }
];

export default function App() {
  const location = useLocation();

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={16} />
          </div>
          <div>
            <p className="brand-title">Anvaya</p>
            <p className="brand-subtitle">Risk Intelligence</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const active = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link key={item.path} to={item.path} className={`nav-item ${active ? "active" : ""}`}>
                <Icon size={16} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <label className="search-box">
            <Search size={16} />
            <input type="text" placeholder="Search customer, case, or transaction..." />
          </label>
          <div className="topbar-right">
            <span className="status-chip">
              <i className="status-indicator" />
              Pipeline: live
            </span>
            <div className="utility-actions">
              <button className="icon-btn" aria-label="Alerts">
                <Bell size={15} />
              </button>
              <button className="icon-btn" aria-label="Settings">
                <Settings size={15} />
              </button>
            </div>
            <div className="profile-pill">
              <UserCircle2 size={18} />
              <div>
                <strong>Jane Smith</strong>
                <small>Risk Investigator</small>
              </div>
            </div>
          </div>
        </header>

        <section className="route-pane">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/pipeline-monitor" element={<PipelineMonitorPage />} />
            <Route path="/party-risk-list" element={<PartyRiskListPage />} />
            <Route path="/explainability" element={<ExplainabilityPage />} />
            <Route path="/model-health" element={<ModelHealthPage />} />
            <Route path="/live-risk-logs" element={<PartyRiskListPage logsOnly />} />
            <Route path="/configuration" element={<ConfigurationPage />} />
          </Routes>
        </section>
      </main>
    </div>
  );
}
