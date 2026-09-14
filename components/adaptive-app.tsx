"use client";

import { useCallback, useState } from "react";
import { Activity, Bell, Blocks, BrainCircuit, Gauge, LayoutDashboard, Search, Settings2, UserRound } from "lucide-react";
import { DashboardView } from "@/components/views/dashboard-view";
import { StudioView } from "@/components/views/studio-view";
import { CapabilitiesView } from "@/components/views/capabilities-view";
import { ProfileView } from "@/components/views/profile-view";
import { SettingsView } from "@/components/views/settings-view";
import type { ViewKey } from "@/lib/platform-data";

const views = {
  dashboard: { title: "工作空间总览", caption: "掌握异构智能体与任务运行状态", icon: LayoutDashboard },
  studio: { title: "自适应编排", caption: "让系统从任务描述自动生成协同流程", icon: Blocks },
  capabilities: { title: "能力空间", caption: "比较、校准并管理异构主体能力", icon: Gauge },
  profile: { title: "用户画像", caption: "依据使用者能力调整系统控制粒度", icon: UserRound },
  settings: { title: "系统设置", caption: "配置模型 API 与执行环境", icon: Settings2 },
};

export function AdaptiveApp() {
  const [active, setActive] = useState<ViewKey>("studio");
  const [toast, setToast] = useState("");
  const CurrentIcon = views[active].icon;
  const notify = useCallback((message: string) => { setToast(message); window.setTimeout(() => setToast(""), 2600); }, []);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><BrainCircuit size={19} strokeWidth={2.3}/></div><div><div className="brand-name">Adaptive-AGES</div><div className="brand-caption">Execution System</div></div></div>
      <div className="nav-label">Workspace</div>
      <nav className="nav-stack" aria-label="主导航">
        {(Object.keys(views) as ViewKey[]).filter(key => key !== "settings").map((key) => { const ItemIcon = views[key].icon; return <button key={key} className={`nav-item ${active === key ? "active" : ""}`} onClick={() => setActive(key)}><ItemIcon size={17}/><span>{views[key].title.replace("工作空间", "")}</span>{key === "studio" && <span className="nav-badge">API</span>}</button>; })}
      </nav>
      <div className="sidebar-bottom"><div className="system-card"><div className="system-line"><span>协同引擎</span><Activity size={14}/></div><div className="system-value"><span className="pulse-dot"/> LangGraph 正常</div></div><button className={`nav-item ${active === "settings" ? "active" : ""}`} onClick={() => setActive("settings")}><Settings2 size={17}/><span>系统设置</span></button></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="page-title"><h1>{views[active].title}</h1><p>{views[active].caption}</p></div><div className="search-box"><Search size={16}/><input aria-label="全局搜索" placeholder="搜索 Agent、任务或工作流"/><span className="key-hint">⌘ K</span></div><div className="top-actions"><button className="icon-button" aria-label="消息"><Bell size={17}/></button><div className="avatar" title="当前用户">PW</div></div></header>
      <div className="content">{active === "dashboard" && <DashboardView onNavigate={() => setActive("studio")}/>} {active === "studio" && <StudioView notify={notify}/>} {active === "capabilities" && <CapabilitiesView notify={notify}/>} {active === "profile" && <ProfileView notify={notify}/>} {active === "settings" && <SettingsView notify={notify}/>}</div>
    </main>
    {toast && <div className="toast"><CurrentIcon size={16}/>{toast}</div>}
  </div>;
}
