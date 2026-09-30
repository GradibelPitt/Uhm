import { ontology, regionLabel } from '../ontology'

interface Props {
  region: string; onRegion: (value: string) => void
  intensity: number; onIntensity: (value: number) => void
  duration: number; onDuration: (value: number) => void
  onPulse: () => void; onReset: () => void
}

export function MockControls({ region, onRegion, intensity, onIntensity, duration, onDuration, onPulse, onReset }: Props) {
  return <div className="mock-panel">
    <div><p className="eyebrow">EXPLICIT MOCK MODE</p><h3>Test a body force event</h3><p>Manual events only. These controls do not run a model.</p></div>
    <div className="mock-controls">
      <label>Body region<select value={region} onChange={event => onRegion(event.target.value)}>
        {ontology.regions.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        {ontology.special_regions.map(id => <option key={id} value={id}>{regionLabel(id)}{id === 'unknown' ? ' / abstain' : ''}</option>)}
      </select></label>
      <label>Intensity <output>{intensity.toFixed(2)}</output><input aria-label="Mock intensity" type="range" min="0" max="1" step="0.01" value={intensity} onChange={event => onIntensity(Number(event.target.value))} /></label>
      <label>Duration <output>{duration} ms</output><input aria-label="Mock duration in milliseconds" type="range" min="20" max="1500" step="20" value={duration} onChange={event => onDuration(Number(event.target.value))} /></label>
      <button className="primary-button" onClick={onPulse}>Send mock event <span aria-hidden="true">↗</span></button>
      <button className="text-button" onClick={onReset}>Reset</button>
    </div>
  </div>
}
