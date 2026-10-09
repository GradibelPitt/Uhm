import { useEffect, useRef, useState } from 'react'
import { HapticPreviewPanel } from './components/HapticPreviewPanel'
import type { ContactReport, ContactState } from './contactPlayback'
import { contactNames, contactsAt, cueNames, feetNames, motionAt, phaseNames, regionNames, regionState, supportsContacts, timecode, validWindow, windowAt } from './contactPlayback'

interface Metadata {
  video_url: string; video_name: string
  video: { fps: number; width: number; height: number; duration_ms: number }
  playback: {
    model: string; request_p50_s: number | null; request_p95_s: number | null
    request_to_step_ratio: number | null; analysis_wall_s: number | null
  }
  source?: { source: string; attribution: string } | null
}
const bodyParts = [
  { id: 'head', path: 'M139 43C139 24 181 24 181 43L179 65Q177 78 160 81Q143 78 141 65Z' },
  { id: 'torso', path: 'M145 81L144 91L122 99Q115 116 126 145L133 180L134 203Q160 216 186 203L187 180L194 145Q205 116 198 99L176 91L175 81Q160 88 145 81Z' },
  { id: 'hips', path: 'M134 207Q160 218 186 207L191 239L175 254L160 243L145 254L129 239Z' },
  { id: 'arms', path: 'M118 101Q108 102 104 119L94 164L83 201L90 209L103 202L115 169L126 140ZM202 101Q212 102 216 119L226 164L237 201L230 209L217 202L205 169L194 140Z' },
  { id: 'hands', path: 'M83 205L76 218L72 243Q76 250 81 241L86 231L86 247Q91 251 96 239L101 219L100 207L90 212ZM237 205L244 218L248 243Q244 250 239 241L234 231L234 247Q229 251 224 239L219 219L220 207L230 212Z' },
  { id: 'legs', path: 'M129 244L145 258L157 248L155 295L150 323L148 374L132 374L125 329L122 294ZM191 244L175 258L163 248L165 295L170 323L172 374L188 374L195 329L198 294Z' },
  { id: 'feet', path: 'M132 379L148 379L148 391L143 401L118 403Q110 396 122 389ZM172 379L188 379L198 389Q210 396 202 403L177 401L172 391Z' },
]
const stateNames: Record<ContactState, string> = { positive: '接触候选', negative: '未检出接触', unknown: '证据不足', unanalysed: '未分析' }
const groups = [
  { title: '足部', actions: ['foot_support', 'foot_push_off', 'foot_landing'] },
  { title: '躯干 / 髋', actions: ['torso_impact', 'torso_support', 'hip_support'] },
  { title: '手 / 臂', actions: ['hand_load', 'hand_impact', 'arm_impact'] },
  { title: '腿 / 头', actions: ['leg_impact', 'head_impact'] },
]

function BodyContactMap({ report, seconds }: { report: ContactReport; seconds: number }) {
  const active = contactsAt(report, seconds)
  return <svg className="contact-body" viewBox="0 0 320 435" role="img" aria-label="人体接触示意图，左右侧未判定">
    <defs>
      <pattern id="contact-grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M24 0H0V24" fill="none" stroke="#dfe5dd" strokeWidth=".6" /></pattern>
      <pattern id="uncertain-body" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#edf0e8" /><path d="M0 0V5" stroke="#c9cbb9" strokeWidth="1.3" /></pattern>
      <linearGradient id="contact-active" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#f9c589" /><stop offset="1" stopColor="#ec8753" /></linearGradient>
    </defs>
    <rect x="36" y="15" width="248" height="401" fill="url(#contact-grid)" />
    <ellipse cx="160" cy="410" rx="66" ry="7" fill="#d9dfd4" opacity=".55" />
    <path d="M160 12V417" stroke="#c1cbbd" strokeDasharray="3 6" />
    {bodyParts.map(part => {
      const state = regionState(report, part.id, seconds)
      const inferred = active.some(event => event.basis === 'inferred' && (event.contact_region === part.id || event.contact_region === 'hands_and_arms' && ['hands', 'arms'].includes(part.id)))
      return <path key={part.id} d={part.path} className={`contact-part ${state}${inferred ? ' inferred' : ''}`} data-region={part.id}
        fill={state === 'positive' ? inferred ? '#7cbeb7' : 'url(#contact-active)' : state === 'unknown' ? 'url(#uncertain-body)' : '#e0e6dc'}
        stroke={state === 'positive' ? '#c6713f' : '#aab7a4'} strokeWidth="1.2">
        <title>{regionNames[part.id]} · {inferred ? '运动推断' : stateNames[state]}</title>
      </path>
    })}
    <path d="M160 101V189M135 138Q160 148 185 138M128 305H153M167 305H192" fill="none" stroke="#a6b49f" strokeWidth=".8" opacity=".65" />
    <g className="body-map-label"><text x="31" y="72">头部</text><path d="M63 69H135" />
      <text x="251" y="164">躯干</text><path d="M193 160H246" />
      <text x="30" y="291">腿部</text><path d="M62 287H119" />
      <text x="247" y="389">足部</text><path d="M202 385H241" /></g>
  </svg>
}

export function ContactWorkbench() {
  const [data, setData] = useState<{ meta: Metadata; report: ContactReport } | null>(null)
  const [error, setError] = useState('')
  const [videoError, setVideoError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [seconds, setSeconds] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [ready, setReady] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loop, setLoop] = useState(false)
  const [presentation, setPresentation] = useState(false)
  const [inspect, setInspect] = useState(false)
  const video = useRef<HTMLVideoElement>(null)
  const shell = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    const read = async (url: string) => {
      const response = await fetch(url, { signal: controller.signal })
      if (!response.ok) throw new Error(`加载失败（${response.status}）`)
      return response.json()
    }
    Promise.all([read('/api/meta'), read('/api/action-timeline')]).then(([meta, report]) => {
      if (!meta.video_url || !meta.playback || !(report.duration_s > 0) || !Array.isArray(report.windows) || !Array.isArray(report.actions)) throw new Error('视频或分析结果不完整')
      setData({ meta, report })
    }).catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '无法连接工作台') })
    return () => controller.abort()
  }, [attempt])
  useEffect(() => {
    if (!playing) return
    let frame: number
    const tick = () => { if (video.current) setSeconds(video.current.currentTime); frame = requestAnimationFrame(tick) }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing])
  const seek = (target: number) => {
    if (!video.current || !data) return
    const next = Math.max(0, Math.min(data.report.duration_s, target))
    if (!playing) video.current.pause()
    video.current.currentTime = next
    setSeconds(next)
  }
  const togglePlay = () => {
    const media = video.current
    if (!media || !ready) return
    if (!media.paused) media.pause()
    else void media.play().catch(() => setVideoError('无法播放视频，请重试或检查浏览器的视频解码支持。'))
  }
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLElement && (event.target.closest('button,input,select,textarea,summary,video,a') || event.target.isContentEditable)) return
      if (event.code === 'Space') { event.preventDefault(); togglePlay() }
      if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') { event.preventDefault(); seek((video.current?.currentTime ?? 0) + (event.code === 'ArrowLeft' ? -1 : 1) / (data?.meta.video.fps ?? 25)) }
      if (event.code === 'Escape') setPresentation(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  })
  if (!data) return <div className="studio-loading"><span className="studio-logo">u<span>h</span>m.</span><h1>{error ? '工作台暂时无法连接' : '正在准备运动工作台'}</h1><p>{error || '加载真实视频与模型分析结果…'}</p>{error && <button onClick={() => setAttempt(value => value + 1)}>重新连接</button>}</div>
  const { meta, report } = data
  const dryRun = report.mode === 'dry_run'
  const pov = report.task === 'egocentric_motion'
  const motion = videoError || !ready ? undefined : motionAt(report, seconds)
  const current = windowAt(report, seconds)
  const active = videoError || !ready ? [] : contactsAt(report, seconds)
  const events = dryRun || !supportsContacts(report) ? [] : (report.contact_events ?? [])
  const tracks = pov ? [{ title: '足部支撑', actions: ['foot_support'] }, { title: '起跳 / 落地', actions: ['foot_push_off', 'foot_landing', 'cushioning'] }, { title: '手部支撑', actions: ['hand_load'] }] : groups
  const pct = (value: number) => `${Math.max(0, Math.min(100, value / report.duration_s * 100))}%`
  const timed = (value: number | null) => value === null ? '未测量' : `${value.toFixed(2)} s`
  const unknown = !current || !validWindow(current) || Object.values(current.labels).some(label => label.state === 'unknown')
  const summary = videoError ? '视频不可用' : !ready ? '等待视频加载' : dryRun ? '尚未运行模型' : !current ? '当前区间未分析' : motion ? feetNames[motion.feet] : active.length ? [...new Set(active.map(event => regionNames[event.contact_region] ?? event.contact_region))].join(' + ') : unknown ? '当前接触证据不足' : '当前未检出接触'
  const displaySeconds = videoError || !ready ? -1 : seconds
  return <div className={`studio-shell ${presentation ? 'presenting' : ''}`} ref={shell}>
    <aside className="studio-rail" aria-label="工作台导航">
      <a className="rail-monogram" href="/" aria-label="UHM 工作台">u<span>.</span></a>
      <a className="rail-item selected" href="#stage" aria-label="视频与人体反馈"><svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="3" /><path d="m10 9 5 3-5 3Z" /></svg></a>
      <a className="rail-item" href="#contact-timeline" aria-label="接触时间轴"><svg viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16M8 4v16" /></svg></a>
      <button className="rail-item" aria-label="模型分析详情" onClick={() => setInspect(value => !value)}><svg viewBox="0 0 24 24"><path d="M5 19V9M12 19V4M19 19v-7" /></svg></button>
      <span className="rail-bottom">UHM<br />LAB</span>
    </aside>
    <div className="studio-main">
      <header className="studio-topbar"><a className="studio-logo" href="/">u<span>h</span>m.</a><span className="product-name">HAPTIC STUDIO</span><span className="topbar-divider" /><span className="project-name">运动感知工作台</span><div className="topbar-actions"><span className="local-indicator"><i />{report.mode === 'local_vlm' ? '本地模型结果' : '视频分析结果'}</span><button className="text-button" onClick={() => setPresentation(value => !value)}>{presentation ? '退出演示' : '演示模式'}<span>↗</span></button></div></header>
      <main>
        <section className="studio-intro"><div><p className="studio-eyebrow">FROM MOTION TO SENSATION</p><h1>让运动，被身体感知<span>。</span></h1><p className="studio-subtitle">第一视角跑酷，跟随奔跑、起跳与落地，观察身体反馈。</p></div><div className="session-label"><span className="session-dot" />SESSION 01<span className="session-sub">视频分析 · 同步回放</span></div></section>
        <div className="studio-notice"><span className="notice-tag">{dryRun ? 'DRY RUN · 未推理' : pov ? 'POV · 运动推断' : '视觉接触候选'}</span><span>{dryRun ? '仅完成视频采样，尚无识别结果。' : pov ? '镜头运动 + 时序线索，推断镜头外的足部支撑。青色为推断，橙色为可见接触；力值未标定。' : '人体反馈随视频时间同步；显示模型推断的接触区域，物理力值与左右侧尚未标定。'}</span><button onClick={() => setInspect(value => !value)} aria-expanded={inspect}>模型详情 <span>↗</span></button></div>
        <section className="studio-stage" id="stage" aria-label="视频与人体同步工作台">
          <article className="video-card studio-card"><div className="studio-card-heading"><h2><span>01</span>运动影像</h2><span className="card-tag">SOURCE VIDEO</span></div>
            <div className="studio-video-wrap"><video ref={video} src={meta.video_url} muted playsInline preload="auto" loop={loop} aria-label="运动源视频"
              onLoadedData={() => { setReady(true); setVideoError(''); if (video.current) video.current.playbackRate = speed }}
              onTimeUpdate={() => setSeconds(video.current?.currentTime ?? 0)} onSeeking={() => setSeconds(video.current?.currentTime ?? 0)}
              onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onSeeked={() => setPlaying(Boolean(video.current && !video.current.paused && !video.current.ended))} onEnded={() => { video.current?.pause(); setPlaying(false); setSeconds(video.current?.currentTime ?? report.duration_s) }}
              onError={() => { setReady(false); setPlaying(false); setVideoError('视频无法解码，请检查源文件。') }} />
              <div className="video-corner"><i />真实录像 <span>{meta.video.width} × {meta.video.height}</span></div>
              {!playing && !videoError && <button className="video-overlay-play" aria-label="播放运动视频" onClick={togglePlay} disabled={!ready}><svg viewBox="0 0 24 24"><path d="m9 5 11 7-11 7Z" /></svg></button>}
              {pov && <div className={`motion-overlay phase-${motion?.phase ?? 'unknown'}`}><span>运动状态</span><strong>{motion ? phaseNames[motion.phase] : '等待分析'}</strong><em>{motion?.basis === 'inferred' ? '运动推断' : motion?.basis === 'visible' ? '可见证据' : '待判断'}</em></div>}
              <div className="video-timecode">{timecode(seconds)}<span> / {timecode(report.duration_s)}</span></div>
              {videoError && <div className="studio-video-error" role="alert">{videoError}<button onClick={() => video.current?.load()}>重新加载</button></div>}
            </div>
            <div className="studio-transport"><div className="transport-row"><button className="transport-play" aria-label={playing ? '暂停' : '播放'} onClick={togglePlay} disabled={!ready}>{playing ? 'Ⅱ' : '▶'}</button><button className="frame-button" aria-label="上一帧" disabled={!ready} onClick={() => { video.current?.pause(); seek(seconds - 1 / meta.video.fps) }}>Ⅰ‹</button><button className="frame-button" aria-label="下一帧" disabled={!ready} onClick={() => { video.current?.pause(); seek(seconds + 1 / meta.video.fps) }}>›Ⅰ</button><span className="transport-time">{timecode(seconds)} <span>/ {timecode(report.duration_s)}</span></span><select aria-label="播放速度" value={speed} onChange={event => { const rate = Number(event.target.value); setSpeed(rate); if (video.current) video.current.playbackRate = rate }}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1.0×</option></select><button className={`loop-button ${loop ? 'enabled' : ''}`} aria-pressed={loop} onClick={() => setLoop(value => !value)}>循环</button></div>
              <input className="video-scrubber" type="range" min="0" max={report.duration_s} step={1 / meta.video.fps} value={seconds} disabled={!ready} onChange={event => seek(Number(event.target.value))} aria-label="视频进度" style={{ background: `linear-gradient(to right, #344e38 ${pct(seconds)}, #dfe4da ${pct(seconds)})` }} />
              <div className="video-file-row"><span className="source-file">{meta.video_name}</span><span>{meta.video.fps.toFixed(0)} FPS<span className="small-divider">/</span>静音视频输入</span></div>
            </div>
          </article>
          <article className="body-card studio-card"><div className="studio-card-heading"><h2><span>02</span>身体反馈</h2><span className="sync-label"><i />媒体时钟同步</span></div>
            <div className="body-visual"><div className="body-view-tag">FRONT VIEW<span>接触区域示意</span></div><BodyContactMap report={report} seconds={displaySeconds} /><div className="body-map-key"><span><i className="key-contact" />可见接触</span>{pov && <span><i className="key-inferred" />运动推断</span>}<span><i className="key-unknown" />证据不足</span></div></div>
            <div className={`body-current ${active.length ? 'has-contact' : ''}`}><span className="current-indicator" /><div><span className="current-eyebrow">当前身体反馈</span><strong>{summary}</strong><p>{motion ? motion.cues.map(cue => cueNames[cue] ?? cue).join(' · ') : active.length ? active.map(event => contactNames[event.event_type] ?? event.event_type).join(' · ') : '等待更多运动线索'}</p></div><span className="current-count">{String(active.length).padStart(2, '0')}</span></div>
            <p className="body-side-note">{pov ? '双侧着色代表足部区域；双脚、单脚与交替支撑以文字判断为准。' : '双侧着色仅作区域示意，不表示两侧均发生接触。'}</p>
          </article>
        </section>
        <section className="timeline-card studio-card" id="contact-timeline"><div className="studio-card-heading"><h2><span>03</span>{pov ? '运动与接触时间轴' : '接触时间轴'} <span className="heading-sub">CONTACT TIMELINE</span></h2><div className="timeline-legend"><span><i className="key-contact" />模型候选</span><span><i className="key-unknown" />未知 / 未分析</span><span className="timeline-hint">点击区间定位视频</span></div></div>
          <div className="timeline-content">{pov && <div className="motion-track-row"><span>运动阶段</span><div className="motion-track">{report.windows.filter(w => !dryRun && validWindow(w) && w.motion).map(w => <button key={w.start_s} className={`motion-segment phase-${w.motion!.phase}`} style={{ left: pct(w.start_s), width: pct(w.end_s - w.start_s) }} onClick={() => seek(w.start_s)} title={`${phaseNames[w.motion!.phase]} · ${w.start_s.toFixed(1)}s`} aria-label={`定位${phaseNames[w.motion!.phase]} ${w.start_s.toFixed(1)}秒`} />)}<span className="contact-playhead" style={{ left: pct(seconds) }} /></div></div>}{pov && <div className="phase-legend">{['running', 'takeoff', 'airborne', 'landing', 'cushioning', 'climbing'].map(phase => <span key={phase}><i className={`phase-${phase}`} />{phaseNames[phase]}</span>)}</div>}{tracks.map(group => <div className="contact-track-row" key={group.title}><span>{group.title}</span><div className="contact-track">
            {report.windows.filter(window => !dryRun && supportsContacts(report) && validWindow(window) && group.actions.every(action => window.labels[action]?.state === 'negative')).map(window => <span key={window.start_s} className="known-range" style={{ left: pct(window.start_s), width: pct(window.end_s - window.start_s) }} />)}
            {events.filter(event => group.actions.includes(event.event_type)).map((event, index) => <button key={`${event.event_type}-${index}`} className={`contact-segment ${event.basis === 'inferred' ? 'inferred' : ''} ${active.includes(event) ? 'is-current' : ''}`} style={{ left: pct(event.start_s), width: pct(event.end_s - event.start_s), top: 2 + group.actions.indexOf(event.event_type) * 13 }} onClick={() => seek(event.start_s)} title={`${contactNames[event.event_type]} · ${event.start_s.toFixed(2)}–${event.end_s.toFixed(2)} s`} aria-label={`定位${contactNames[event.event_type]} ${event.start_s.toFixed(2)}秒`}><span>{contactNames[event.event_type]}</span></button>)}
            <span className="contact-playhead" style={{ left: pct(seconds) }} /></div></div>)}
            <div className="timeline-ticks">{Array.from({ length: 5 }, (_, index) => <span key={index}>{(report.duration_s * index / 4).toFixed(2)}s</span>)}</div></div>
        </section>
        <section className="studio-metrics" aria-label="模型分析概况"><div><span>ANALYSIS ENGINE</span><strong>{report.model}</strong><small>已保存的视觉模型分析结果</small></div><div><span>TEMPORAL RESOLUTION</span><strong>{(report.step_s * 1000).toFixed(0)} <em>ms</em></strong><small>状态更新间隔，非逐帧事件精度</small></div><div><span>INFERENCE P50</span><strong>{timed(meta.playback.request_p50_s)}</strong><small>{pov ? '每批请求耗时，非显示延迟' : '每窗口请求耗时，非显示延迟'}</small></div><div><span>FORCE MAGNITUDE</span><strong className="unmeasured">未标定 <em>↗</em></strong><small>当前无牛顿值、强度或方向输出</small></div></section>
        {inspect && <section className="studio-inspector studio-card" aria-label="模型分析详情"><div className="studio-card-heading"><h2>分析证据与能力边界</h2><button className="text-button" onClick={() => setInspect(false)}>收起 −</button></div><div className="inspector-grid"><div><h3>同步回放 ≠ 实时推理</h3><p>这些结果由模型预先分析，播放时按视频时间读取。P95 请求耗时 {timed(meta.playback.request_p95_s)}。{meta.playback.request_to_step_ratio !== null && `中位请求耗时为采样步长的 ${meta.playback.request_to_step_ratio.toFixed(1)} 倍。`}</p><p>当前没有独立真实运动测试集的精度验收，也没有物理受力标定。</p></div><div><h3>当前窗口 {current ? `${current.start_s.toFixed(2)}–${current.end_s.toFixed(2)} s` : '未分析'}</h3><div className="evidence-list">{current && report.actions.map(action => <div key={action}><span>{contactNames[action] ?? action}</span><span>{dryRun || !validWindow(current) ? '未分析 / 错误' : stateNames[current.labels[action]?.state ?? 'unknown']}</span>{current.labels[action]?.evidence_times_s?.map((stamp, index) => <button key={stamp} onClick={() => seek(stamp)}>{index === 0 ? '变化前' : '变化后'} {stamp.toFixed(2)}s ↗</button>)}</div>)}</div></div></div></section>}
        <HapticPreviewPanel seconds={displaySeconds} onSeek={seek} />
        <footer className="studio-footer"><span>UHM <i>/</i> VISUAL CONTACT RESEARCH</span><span>空格 播放 / 暂停 <i>·</i> ← → 逐帧</span></footer>
        {meta.source && <details className="source-attribution"><summary>素材来源与署名</summary><p><a href={meta.source.source} target="_blank" rel="noreferrer">查看原始来源 ↗</a> · {meta.source.attribution}</p></details>}
      </main>
    </div>
  </div>
}
