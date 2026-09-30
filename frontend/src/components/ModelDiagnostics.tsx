import type { Diagnostics, RuntimeMetadata } from '../useInference'
import { regionLabel } from '../ontology'

const percent = (value: number | undefined) => typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—'

export function ModelDiagnostics({ diagnostics: result, metadata, playing }: { diagnostics: Diagnostics | null; metadata: RuntimeMetadata | null; playing: boolean }) {
  const outputs = [
    ['Visual event', result?.event_class.replaceAll('_', ' ') ?? 'Awaiting frames'],
    ['Recipient', result?.recipient.replaceAll('_', ' ') ?? '—'],
    ['Body region', result ? regionLabel(result.body_region) : '—'],
    ['Perceptual intensity', result?.intensity?.toFixed(2) ?? '—'],
    ['Event confidence', percent(result?.confidence?.event)],
    ['Region observable', percent(result?.observable_probability)],
  ]
  return <div className="model-panel">
    <div className="model-panel-heading"><span className="eyebrow">RAW MODEL OUTPUT</span><span className="event-state"><span className={`status-dot ${playing ? '' : 'neutral'}`} />{playing ? 'Sampling video frames' : 'Paused · press play to sample'}</span></div>
    <div className="diagnostics">
      {outputs.map(([label, value]) => <div key={label}><div className="diagnostic-label">{label}</div><div className="diagnostic-value">{value}</div></div>)}
      <p className="diagnostic-note">Raw predictions are shown before event gating. Only accepted player events with a known region light the body map. Unknown is an abstention.</p>
    </div>
    <div className="runtime-meta"><span>{metadata?.model.name ?? 'Model unavailable'}</span><span>{metadata?.model.device.toUpperCase() ?? '—'}</span><span>SCHEMA {metadata?.schema_version ?? '—'}</span></div>
    {metadata?.model.synthetic && <p className="training-note">Model training: synthetic fixtures only. Pipeline validation; not evidence of gameplay accuracy.</p>}
  </div>
}
