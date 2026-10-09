import { useEffect, useState } from 'react'
import { contactNames, regionNames } from '../contactPlayback'
import { envelopeAt } from '../hapticPreview'
import type { HapticPreview } from '../hapticPreview'
import './hapticPreview.css'

export function HapticPreviewPanel({ seconds, onSeek }: { seconds: number; onSeek: (time: number) => void }) {
  const [preview, setPreview] = useState<HapticPreview | null>(null)
  const [includeInferred, setIncludeInferred] = useState(false)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [fullVideo, setFullVideo] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    setPreview(null)
    setError('')
    fetch(`/api/haptic-preview?include_inferred=${includeInferred}`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error('触觉预览暂不可用，请确认后端已更新。')
        const value = await response.json() as HapticPreview
        if (value.mode !== 'design_preview' || value.amplitude_semantics !== 'unit_envelope_not_force'
          || value.schema_version !== '0.1' || !Number.isFinite(value.duration_ms) || value.duration_ms <= 0
          || !Array.isArray(value.cues)) throw new Error('触觉预览格式不受支持。')
        if (!controller.signal.aborted) setPreview(value)
      })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : '预览加载失败') })
    return () => controller.abort()
  }, [includeInferred, attempt])
  const tracks = [...new Map(preview?.cues.map(cue => [`${cue.region}:${cue.effect}`, { region: cue.region, effect: cue.effect }]) ?? []).values()]
  const duration = preview?.duration_ms ?? 1
  const rangeStart = fullVideo ? 0 : Math.max(0, Math.min(Math.max(0, seconds) * 1000 - 3000, duration - 6000))
  const rangeEnd = fullVideo ? duration : Math.min(duration, rangeStart + 6000)
  const x = (time: number) => (time - rangeStart) / (rangeEnd - rangeStart) * 1000
  return <section className="haptic-preview studio-card" aria-label="触觉时序预览">
    <div className="studio-card-heading"><h2>触觉时序预览 <span className="heading-sub">ENVELOPE PREVIEW</span></h2>
      <label><input type="checkbox" checked={includeInferred} onChange={event => setIncludeInferred(event.target.checked)} /> 纳入运动推断</label></div>
    <p>等幅设计预览：碰撞为短脉冲，支撑为持续包络。曲线高度不是受力强度；尚未输出到硬件。</p>
    {preview && <div className="haptic-preview-range"><span>{(rangeStart / 1000).toFixed(2)}–{(rangeEnd / 1000).toFixed(2)} s</span><button aria-pressed={fullVideo} onClick={() => setFullVideo(value => !value)}>{fullVideo ? '查看当前 6 秒' : '查看完整视频'}</button></div>}
    {error ? <p role="status">{error} <button onClick={() => setAttempt(value => value + 1)}>重试预览</button></p>
      : !preview ? <p role="status">加载触觉预览…</p>
        : !tracks.length ? <p>当前没有符合条件的接触；可选择纳入运动推断。</p>
          : tracks.map(({ region, effect }) => {
            const cues = preview.cues.filter(cue => cue.region === region && cue.effect === effect)
            const name = `${regionNames[region] ?? region} · ${effect === 'pulse' ? '脉冲' : '持续'}`
            const level = Math.max(0, ...cues.map(cue => envelopeAt(cue, seconds * 1000)))
            return <div className="haptic-preview-row" key={`${region}:${effect}`}>
            <span>{name}</span>
            <svg viewBox="0 0 1000 48" preserveAspectRatio="none" role="group" aria-label={`${name}触觉包络`}>
              <path d="M0 44H1000" className="envelope-baseline" />
              {cues.filter(cue => cue.start_ms < rangeEnd && cue.end_ms > rangeStart).map(cue => {
                const points = cue.envelope.map(point => `${x(cue.start_ms + point.offset_ms)},${44 - point.level * 36}`).join(' ')
                const label = `${contactNames[cue.event_type] ?? cue.event_type} · ${(cue.start_ms / 1000).toFixed(2)}s · ${cue.basis === 'inferred' ? '运动推断' : '接触候选'}`
                return <g key={cue.cue_id} className={cue.basis === 'inferred' ? 'envelope-inferred' : 'envelope-visible'}>
                  <title>{label}</title><polyline points={points} />
                  <rect x={Math.max(0, x(cue.start_ms))} y="0" width={Math.max(1, x(Math.min(cue.end_ms, rangeEnd)) - Math.max(0, x(cue.start_ms)))} height="48"
                    fill="transparent" role="button" tabIndex={0} aria-label={`定位触觉 ${label}`}
                    onClick={() => onSeek(cue.start_ms / 1000)}
                    onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); onSeek(cue.start_ms / 1000) } }} />
                </g>
              })}
              {seconds >= 0 && <path d={`M${Math.min(1000, x(seconds * 1000))} 0v48`} className="envelope-cursor" />}
            </svg>
            <meter min={0} max={1} value={level} aria-label={`${name}当前包络`} />
          </div>})}
  </section>
}
