"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, KeyRound, LoaderCircle, PlugZap, Save, ShieldCheck } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface ModelSettings {
  provider: string;
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  api_key_configured: boolean;
}

const providers = [
  { value:"openai", label:"OpenAI", baseUrl:"", model:"gpt-4.1-mini", placeholder:"https://api.openai.com/v1", help:"OpenAI 官方接口，Base URL 可留空。" },
  { value:"newapi", label:"NewAPI / OneAPI 转接", baseUrl:"", model:"gpt-4.1-mini", placeholder:"https://你的网关域名/v1", help:"填写购买服务时提供的 API 网关，不要填写文档站地址。" },
  { value:"deepseek", label:"DeepSeek", baseUrl:"https://api.deepseek.com/v1", model:"deepseek-v4-flash", placeholder:"https://api.deepseek.com/v1", help:"DeepSeek 官方 OpenAI-compatible 接口。" },
  { value:"bailian-cn", label:"阿里云百炼（中国内地）", baseUrl:"https://dashscope.aliyuncs.com/compatible-mode/v1", model:"qwen-plus", placeholder:"https://dashscope.aliyuncs.com/compatible-mode/v1", help:"阿里云百炼中国内地兼容接口。" },
  { value:"bailian-intl", label:"阿里云百炼（国际）", baseUrl:"https://dashscope-intl.aliyuncs.com/compatible-mode/v1", model:"qwen-plus", placeholder:"https://dashscope-intl.aliyuncs.com/compatible-mode/v1", help:"阿里云百炼国际兼容接口。" },
  { value:"openai-compatible", label:"自定义 OpenAI-compatible", baseUrl:"", model:"gpt-4.1-mini", placeholder:"https://api.example.com/v1", help:"适用于第三方代理、兼容网关和自托管服务。" },
  { value:"local", label:"本地 Ollama / vLLM", baseUrl:"http://host.docker.internal:11434/v1", model:"qwen2.5:7b", placeholder:"http://host.docker.internal:11434/v1", help:"Docker 访问宿主机通常使用 host.docker.internal。" },
] as const;

const initial: ModelSettings = {provider:"openai-compatible",model:"gpt-4.1-mini",base_url:"",api_key:"",temperature:.2,api_key_configured:false};

export function SettingsView({notify}:{notify:(message:string)=>void}) {
  const [settings,setSettings] = useState(initial);
  const [loading,setLoading] = useState(true);
  const [saving,setSaving] = useState(false);
  const [testing,setTesting] = useState(false);
  const preset = useMemo(() => providers.find(item => item.value === settings.provider) || providers[5], [settings.provider]);
  const invalidDocsUrl = settings.base_url.toLowerCase().includes("docs.newapi.pro");

  useEffect(() => { void apiFetch<Omit<ModelSettings,"api_key">>("/settings/model").then(data => setSettings(current => ({...current,...data,base_url:data.base_url || ""}))).catch(error => notify(error.message)).finally(() => setLoading(false)); }, [notify]);
  const change = (key:keyof ModelSettings,value:string|number|boolean) => setSettings(current => ({...current,[key]:value}));
  const changeProvider = (provider:string) => {
    const next = providers.find(item => item.value === provider);
    setSettings(current => ({...current,provider,base_url:next?.baseUrl || "",model:next?.model || current.model}));
  };
  const save = async (silent = false): Promise<boolean> => {
    if (invalidDocsUrl) { notify("docs.newapi.pro 是文档站，不是 API 网关；请填写服务商实际提供的 /v1 地址"); return false; }
    setSaving(true);
    try {
      const result = await apiFetch<{api_key_configured:boolean}>("/settings/model", {method:"PUT",body:JSON.stringify({provider:settings.provider,model:settings.model.trim(),base_url:settings.base_url.trim() || null,api_key:settings.api_key || null,temperature:settings.temperature})});
      setSettings(current => ({...current,api_key:"",base_url:current.base_url.replace(/\/+$/, ""),api_key_configured:result.api_key_configured}));
      if (!silent) notify("模型 API 设置已加密保存");
      return true;
    } catch (error) { notify(error instanceof Error ? error.message : "保存失败"); return false; }
    finally { setSaving(false); }
  };
  const saveAndTest = async () => {
    setTesting(true);
    try {
      if (!await save(true)) return;
      const result = await apiFetch<{message:string}>("/settings/model/test",{method:"POST"});
      notify(`配置已保存，连接成功：${result.message}`);
    } catch (error) { notify(error instanceof Error ? error.message : "连接失败"); }
    finally { setTesting(false); }
  };

  return <>
    <div className="section-heading"><div><p className="eyebrow">System Settings</p><h2>模型与执行设置</h2><p>设置平台默认模型；每个 LLM 节点也可以覆盖为独立模型。</p></div><button className="primary-button" onClick={() => void save()} disabled={loading || saving}><Save size={14}/>保存设置</button></div>
    <div className="settings-grid"><section className="panel settings-form"><div className="panel-header"><div><h3>默认模型 API</h3><p>API Key 在后端加密存储，不会返回浏览器</p></div>{settings.api_key_configured ? <span className="status-pill live"><CheckCircle2 size={12}/>已配置</span> : <span className="status-pill review">未配置</span>}</div><div className="settings-fields">
      <label><span>接口类型</span><select className="field-input" value={settings.provider} onChange={event => changeProvider(event.target.value)}>{providers.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
      <label><span>模型名称</span><input className="field-input" value={settings.model} onChange={event => change("model",event.target.value)} placeholder={preset.model}/></label>
      <label className="wide"><span>Base URL</span><input className="field-input" value={settings.base_url} onChange={event => change("base_url",event.target.value)} placeholder={preset.placeholder}/><small className="field-help">{preset.help}</small></label>
      {invalidDocsUrl && <div className="provider-warning wide"><AlertTriangle size={15}/><span><strong>地址无效：</strong>https://docs.newapi.pro 是产品文档站。请填写你的第三方服务商或自部署 NewAPI 的实际网关，例如 https://你的域名/v1。</span></div>}
      <label className="wide"><span>API Key {settings.api_key_configured && "（留空表示保持现有 Key）"}</span><div className="secret-input"><KeyRound size={14}/><input type="password" value={settings.api_key} onChange={event => change("api_key",event.target.value)} placeholder={settings.api_key_configured ? "••••••••••••••••" : "sk-..."}/></div></label>
      <label className="wide"><span>Temperature：{settings.temperature.toFixed(1)}</span><input type="range" min="0" max="2" step="0.1" value={settings.temperature} onChange={event => change("temperature",Number(event.target.value))}/></label>
      <div className="settings-actions wide"><button className="ghost-button" onClick={() => void saveAndTest()} disabled={testing || saving || invalidDocsUrl}>{testing ? <LoaderCircle className="spin" size={13}/> : <PlugZap size={13}/>}保存并测试</button><button className="primary-button" onClick={() => void save()} disabled={saving || invalidDocsUrl}>{saving ? <LoaderCircle className="spin" size={13}/> : <Save size={13}/>}保存</button></div>
    </div></section><aside className="settings-note"><ShieldCheck size={19}/><strong>安全与生效范围</strong><p>Key 使用由 JWT_SECRET 派生的密钥加密后存入 PostgreSQL。修改 JWT_SECRET 会使旧 Key 无法解密，需要重新保存。</p><p>这里是自动规划和 LLM 节点的默认配置。节点选择“独立模型”后，将优先使用该节点自己的接口、模型和 Key。</p></aside></div>
  </>;
}
