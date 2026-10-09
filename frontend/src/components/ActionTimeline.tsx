import { useEffect, useState } from 'react'

type Label = { state: 'positive' | 'negative' | 'unknown'; probability: number | null }
type Window = { start_s: number; end_s: number; status: string; labels: Record<string, Label> }
type Segment = { label: string; start_s: number; end_s: number; probability: number | null }
type Timeline = { duration_s: number; actions: string[]; segments: Segment[]; windows: Window[]; model: string; notes: string }

const time = (seconds: number) => `${seconds.toFixed(2)}s`

export function ActionTimeline({ playbackSeconds, onSeek }: { playbackSeconds: number; onSeek: (seconds: number) => void }) {
  const [timeline, setTimeline] = useState<Timeline | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    fetch('/api/action-timeline').then(async response => {
      if (response.status === 404) return null
      if (!response.ok) throw new Error(`Action timeline request failed (${response.status})`)
      return response.json() as Promise<Timeline>
    }).then(data => { if (!cancelled) setTimeline(data) })
      .catch(reason => { if (!cancelled) setError(String(reason)) })
    return () => { cancelled = true }
  }, [])
  if (error) return <p className="action-error" role="alert">{error}</p>
  if (!timeline) return null
  const duration = Math.max(timeline.duration_s, 0.001)
  const percent = (seconds: number) => `${Math.min(100, Math.max(0, seconds / duration * 100))}%`
  const unresolved = timeline.windows.filter(window => window.status !== 'ok')
  return <section className="action-panel" aria-label="Video action timeline">
    <div className="panel-heading"><h2><span className="panel-number">03</span> Action timeline</h2><span className="mono muted">{timeline.model} · OFFLINE</span></div>
    <p className="action-note">{timeline.notes} Click a bar to seek the video.</p>
    <div className="action-rows">
      {timeline.actions.map(action => <div className="action-row" key={action}>
        <span className="action-name">{action.replaceAll('_', ' ')}</span>
        <div className="action-track" aria-label={`${action} time track`}>
          {timeline.windows.filter(window => window.labels[action]?.state === 'unknown').map(window =>
            <span key={`${window.start_s}-${window.end_s}`} className="action-unknown" title={`${time(window.start_s)}–${time(window.end_s)}: ${window.status}`}
              style={{ left: percent(window.start_s), width: percent(window.end_s - window.start_s) }} />)}
          {timeline.segments.filter(segment => segment.label === action).map(segment =>
            <button key={`${segment.start_s}-${segment.end_s}`} className="action-segment" style={{ left: percent(segment.start_s), width: percent(segment.end_s - segment.start_s) }}
              title={`${action}: ${time(segment.start_s)}–${time(segment.end_s)} · ${segment.probability === null ? 'categorical label' : 'p=' + segment.probability.toFixed(2)}`}
              aria-label={`${action} from ${time(segment.start_s)} to ${time(segment.end_s)}`} onClick={() => onSeek(segment.start_s)} />)}
          <span className="action-playhead" style={{ left: percent(playbackSeconds) }} aria-hidden="true" />
        </div>
      </div>)}
    </div>
    <div className="action-axis"><span>0s</span><span>{time(duration)}</span></div>
    <p className="action-summary">{timeline.segments.length} labeled intervals · {unresolved.length} uncertain/error windows</p>
    {unresolved.length > 0 && <details className="action-details"><summary>Review uncertain windows</summary><ul>{unresolved.map(window =>
      <li key={window.start_s}><button onClick={() => onSeek(window.start_s)}>{time(window.start_s)}–{time(window.end_s)}</button> · {window.status}</li>)}</ul></details>}
  </section>
}
