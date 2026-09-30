import type { BodyForceEvent } from '../event'
import { regionLabel } from '../ontology'

export function EventLog({ events, mock }: { events: BodyForceEvent[]; mock: boolean }) {
  const download = () => {
    const url = URL.createObjectURL(new Blob([events.map(event => JSON.stringify(event)).join('\n') + '\n'], { type: 'application/x-ndjson' }))
    const link = document.createElement('a')
    link.href = url
    link.download = `${mock ? 'mock' : 'inference'}-body-force-events.jsonl`
    link.click()
    URL.revokeObjectURL(url)
  }
  return <section className="event-section">
    <div className="event-heading">
      <div><p className="eyebrow">STRUCTURED OUTPUT</p><h2>Body force events <span>{events.length.toString().padStart(2, '0')}</span></h2></div>
      <div className="event-heading-right"><span className="mono muted">{mock ? 'MOCK EVENTS' : 'MODEL EVENTS'} · LAST 100</span><button className="event-export" onClick={download} disabled={!events.length}>Export JSONL ↓</button></div>
    </div>
    <div className="event-table-wrap"><table>
      <thead><tr><th>Time</th><th>Event</th><th>Recipient</th><th>Body region</th><th>Intensity</th><th>Duration</th><th>Confidence</th></tr></thead>
      <tbody>{events.length === 0 ? <tr><td colSpan={7} className="empty-events">{mock ? 'No events yet. Send a mock event to inspect its structured representation.' : 'No events yet. Play the video to run the visual model.'}</td></tr> : [...events].reverse().map(event => <tr key={event.event_id}>
        <td className="mono">{(event.timestamp_ms / 1000).toFixed(2)} s</td><td><span className="event-pill">{event.event_class}</span></td><td>{event.recipient.replaceAll('_', ' ')}</td><td>{regionLabel(event.body_region)}</td><td className="mono">{event.intensity.toFixed(2)}</td><td className="mono">{event.duration_ms} ms</td><td className="mono" title={`Event: ${event.confidence.event.toFixed(3)} · Recipient: ${event.confidence.recipient.toFixed(3)} · Region: ${event.confidence.region.toFixed(3)} · Intensity: ${event.confidence.intensity.toFixed(3)}`}>{(event.confidence.event * 100).toFixed(0)}%</td>
      </tr>)}</tbody>
    </table></div>
  </section>
}
