"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, KeyRound, LoaderCircle, PlugZap, Save, ShieldCheck } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface ModelSettings {
  provider: string;
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  api_key_configured: boolean;
}

const initial: ModelSettings = {provider:"openai-compatible",model:"gpt-4.1-mini",base_url:"",api_key:"",temperature:.2,api_key_configured:false};

export function SettingsView({notify}:{notify:(message:string)=>void}) {
  const [settings,setSettings] = useState(initial);
  const [loading,setLoading] = useState(true);
  const [testing,setTesting] = useState(false);

  useEffect(() => { void apiFetch<Omit<ModelSettings,"api_key">>("/settings/model").then(data => setSettings(current => ({...current,...data,base_url:data.base_url || ""}))).catch(error => notify(error.message)).finally(() => setLoading(false)); }, [notify]);
  const change = (key:keyof ModelSettings,value:string|number|boolean) => setSettings(current => ({...current,[key]:value}));
  const save = async () => {
    try {
      const result = await apiFetch<{api_key_configured:boolean}>("/settings/model", {method:"PUT",body:JSON.stringify({provider:settings.provider,model:settings.model,base_url:settings.base_url || null,api_key:settings.api_key || null,temperature:settings.temperature})});
      setSettings(current => ({...current,api_key:"",api_key_configured:result.api_key_configured})); notify("模型 API 设置已加密保存");
    } catch (error) { notify(error instanceof Error ? error.message : "保存失败"); }
  };
  const test = async () => {
    setTesting(true);
    try { const result = await apiFetch<{message:string}>("/settings/model/test",{method:"POST"}); notify(`连接成功：${result.message}`); }
    catch (error) { notify(error instanceof Error ? error.message : "连接失败"); }
    finally { setTesting(false); }
  };

  return <>
    <div className="section-heading"><div><p className="eyebrow">System Settings</p><h2>模型与执行设置</h2><p>配置 OpenAI API 兼容接口；保存后自动规划和 LLM 节点会使用该模型。</p></div><button className="primary-button" onClick={save} disabled={loading}><Save size={14}/>保存设置</button></div>
    <div className="settings-grid"><section className="panel settings-form"><div className="panel-header"><div><h3>模型 API</h3><p>API Key 在后端加密存储，不会返回浏览器</p></div>{settings.api_key_configured ? <span className="status-pill live"><CheckCircle2 size={12}/>已配置</span> : <span className="status-pill review">未配置</span>}</div><div className="settings-fields">
      <label><span>接口类型</span><select className="field-input" value={settings.provider} onChange={event => change("provider",event.target.value)}><option value="openai-compatible">OpenAI Compatible</option><option value="openai">OpenAI</option><option value="azure-openai">Azure OpenAI Compatible</option><option value="local">Local / vLLM / Ollama</option></select></label>
      <label><span>模型名称</span><input className="field-input" value={settings.model} onChange={event => change("model",event.target.value)} placeholder="gpt-4.1-mini"/></label>
      <label className="wide"><span>Base URL（OpenAI 官方接口可留空）</span><input className="field-input" value={settings.base_url} onChange={event => change("base_url",event.target.value)} placeholder="https://api.openai.com/v1"/></label>
      <label className="wide"><span>API Key {settings.api_key_configured && "（留空表示保持现有 Key）"}</span><div className="secret-input"><KeyRound size={14}/><input type="password" value={settings.api_key} onChange={event => change("api_key",event.target.value)} placeholder={settings.api_key_configured ? "••••••••••••••••" : "sk-..."}/></div></label>
      <label className="wide"><span>Temperature：{settings.temperature.toFixed(1)}</span><input type="range" min="0" max="2" step="0.1" value={settings.temperature} onChange={event => change("temperature",Number(event.target.value))}/></label>
      <div className="settings-actions wide"><button className="ghost-button" onClick={test} disabled={testing || !settings.api_key_configured}>{testing ? <LoaderCircle className="spin" size={13}/> : <PlugZap size={13}/>}测试连接</button><button className="primary-button" onClick={save}><Save size={13}/>保存</button></div>
    </div></section><aside className="settings-note"><ShieldCheck size={19}/><strong>安全与生效范围</strong><p>Key 使用由 JWT_SECRET 派生的密钥加密后存入 PostgreSQL。修改 JWT_SECRET 会使旧 Key 无法解密，需要重新保存。</p><p>模型用于任务拆分和 LLM 节点执行；未配置或调用失败时，任务拆分自动降级为规则规划。</p></aside></div>
  </>;
}
