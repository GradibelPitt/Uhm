import { useCallback, useRef, useState } from 'react'
import { BodyMap } from './body-renderer/BodyMap'
import { isBodyForceEvent } from './event'
import type { BodyForceEvent, BodyRegion } from './event'
import { isBodyRegion, regionLabel } from './ontology'
import { MockControls } from './components/MockControls'
import { EventLog } from './components/EventLog'
import { PageHeader } from './components/PageHeader'
import { ModelDiagnostics } from './components/ModelDiagnostics'
import { useInference } from './useInference'

type Mode = 'inference' | 'mock'
const milliseconds = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(0)} ms` : '—'

export default function App() {
  const [mode, setMode] = useState<Mode>('inference')
  const [region, setRegion] = useState<BodyRegion>('left_torso')
  const [intensity, setIntensity] = useState(0.74)
  const [duration, setDuration] = useState(220)
  const [mockEvents, setMockEvents] = useState<BodyForceEvent[]>([])
  const [latestMockEvent, setLatestMockEvent] = useState<BodyForceEvent | null>(null)
  const [mockResetKey, setMockResetKey] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [videoError, setVideoError] = useState<string | null>(null)
  const [videoReady, setVideoReady] = useState(false)
  const video = useRef<HTMLVideoElement>(null)
  const count = useRef(0)
  const runtime = useInference(video, mode === 'inference')
  const mediaClock = useCallback(() => (video.current?.currentTime ?? 0) * 1000, [])
  const selectRegion = (value: string) => { if (isBodyRegion(value)) setRegion(value) }
  const resetMock = () => { setMockEvents([]); setLatestMockEvent(null); setMockResetKey(value => value + 1) }
  const pulse = () => {
    const event: BodyForceEvent = {
      event_id: `mock_${++count.current}`, timestamp_ms: Math.round(mediaClock()), event_class: 'impact', recipient: 'player', body_region: region,
      intensity, duration_ms: duration, direction: 'unknown', confidence: { event: 1, recipient: 1, region: 1, intensity: 1 },
    }
    if (isBodyForceEvent(event)) {
      setMockEvents(current => [...current.slice(-99), event])
      setLatestMockEvent(event)
      console.info('[BodyForceEvent / explicit mock]', event)
    }
  }
  const changeMode = (next: Mode) => {
    if (next === mode) return
    video.current?.pause()
    runtime.reset()
    setLatestMockEvent(null)
    setMockResetKey(value => value + 1)
    setMode(next)
  }
  const togglePlayback = () => {
    const media = video.current
    if (!media) return
    if (media.paused) void media.play().catch(() => setVideoError('Video playback could not start. Use the video controls to retry.'))
    else media.pause()
  }
  const inference = mode === 'inference'
  const events = inference ? runtime.events : mockEvents
  const lastEvent = inference ? runtime.latestEvent : latestMockEvent
  const modelReady = runtime.connection === 'ready'
  const sourceViewpoint = runtime.metadata?.video.viewpoint === 'third_person' ? 'Third-person · front view' : runtime.metadata?.video.viewpoint === 'first_person' ? 'First-person' : 'Viewpoint unspecified'
  const status = !inference ? 'Manual mock mode · no model inference' : modelReady ? (playing ? 'Model inference active' : 'Model connected · ready to play') : runtime.connection === 'connecting' ? 'Connecting to model runtime…' : 'Model inference unavailable'
  const bodyCaption = !lastEvent ? 'Awaiting a body force event' : lastEvent.recipient !== 'player' ? 'Other recipient · no body response' : lastEvent.body_region === 'unknown' ? 'Unknown region · abstained' : `Latest: ${regionLabel(lastEvent.body_region)} · ${lastEvent.intensity.toFixed(2)}`

  return <div className="app-shell">
    <PageHeader />
    <main>
      <div className="workspace-bar">
        <div className="workspace-label"><span className="live-square" />LIVE WORKSPACE</div>
        <div className="mode-bar">
          {runtime.metadata?.model.synthetic && <span className="synthetic-tag">SYNTHETIC BASELINE</span>}
          <div className="segmented" role="group" aria-label="Event source">
            <button aria-pressed={inference} onClick={() => changeMode('inference')}>Model inference</button>
            <button aria-pressed={!inference} onClick={() => changeMode('mock')}>Manual mock</button>
          </div>
        </div>
      </div>
      <section className="workspace">
        <div className="video-column">
          <div className="panel-heading"><h2><span className="panel-number">01</span> Source video</h2><span className="source-context"><span className="mono muted">PRERECORDED INPUT</span>{runtime.metadata && <span className="source-viewpoint">{sourceViewpoint}</span>}</span></div>
          {runtime.metadata?.video.synthetic && <div className="synthetic-video-banner">SYNTHETIC VIDEO — PIPELINE TEST</div>}
          <div className="video-stage">
            {runtime.metadata ? <video
              ref={video} src={runtime.metadata.video_url} controls muted playsInline preload="auto"
              aria-label="Source video for visual model inference" controlsList="nodownload noremoteplayback" disablePictureInPicture
              onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
              onLoadedData={() => { setVideoReady(true); setVideoError(null); if (video.current) video.current.playbackRate = playbackRate }}
              onSeeking={runtime.reset} onRateChange={() => { setPlaybackRate(video.current?.playbackRate ?? 1); runtime.reset() }}
              onEnded={() => { setPlaying(false); runtime.reset() }}
              onVolumeChange={() => { if (video.current) video.current.muted = true }}
              onError={() => { setVideoReady(false); setVideoError('The source video could not be decoded. Verify the runtime video file and browser codec support.') }}
            /> : <div className="video-empty">
              <svg width="42" height="42" viewBox="0 0 42 42" aria-hidden="true"><rect x="5" y="8" width="32" height="26" rx="3" fill="none" stroke="currentColor" /><path d="m18 15 10 6-10 6Z" fill="currentColor" /></svg>
              <strong>{runtime.error ? 'Visual input unavailable' : 'Connecting to the visual model'}</strong>
              <span>{runtime.error ? 'Start the local runtime, then reconnect.' : 'Loading the configured video and checkpoint.'}</span>
            </div>}
            {videoError && <div className="video-error" role="alert">{videoError}</div>}
          </div>
          <div className="source-footer">
            <span className={`status-dot ${inference && !modelReady ? (runtime.error ? 'error' : 'neutral') : !inference ? 'neutral' : ''}`} />
            <span role="status">{status}</span><span className="mono">{runtime.metadata?.video.fps?.toFixed(0) ?? '—'} SOURCE FPS · NO AUDIO</span>
          </div>
          <div className="source-action-row">
            <span className="source-name" title={runtime.metadata?.video_name}>{runtime.metadata?.video_name ?? 'No source loaded'}</span>
            <label className="playback-speed">SPEED <select aria-label="Video playback speed" value={playbackRate} onChange={event => { const rate = Number(event.target.value); setPlaybackRate(rate); if (video.current) video.current.playbackRate = rate }} disabled={!videoReady}>
              <option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="1.5">1.5×</option><option value="2">2×</option>
            </select></label>
            <button className="primary-button play-button" onClick={togglePlayback} disabled={!videoReady || (inference && !modelReady)} aria-label={playing ? 'Pause source video' : 'Play source video'}><span aria-hidden="true">{playing ? 'Ⅱ' : '▶'}</span>{playing ? 'Pause video' : 'Play video'}</button>
          </div>
          {inference && runtime.error && <div className="error-note" role="alert">{runtime.error}<button className="connection-retry" onClick={runtime.retry}>Reconnect</button></div>}
          {inference ? <>
            <ModelDiagnostics diagnostics={runtime.diagnostics} metadata={runtime.metadata} playing={playing && modelReady} />
            <div className="stat-strip">
              <div><div className="stat-label">SAMPLED WINDOWS</div><div className="stat-value">{runtime.stats.sampled_windows ?? 0}</div></div>
              <div><div className="stat-label">PIPELINE P50</div><div className="stat-value">{milliseconds(runtime.stats.pipeline_ms?.p50)}</div></div>
              <div><div className="stat-label">PIPELINE P95</div><div className="stat-value">{milliseconds(runtime.stats.pipeline_ms?.p95)}</div></div>
            </div>
          </> : <MockControls region={region} onRegion={selectRegion} intensity={intensity} onIntensity={setIntensity} duration={duration} onDuration={setDuration} onPulse={pulse} onReset={resetMock} />}
        </div>
        <aside className="body-column">
          <div className="panel-heading"><h2><span className="panel-number">02</span> Body response</h2><span className="mono muted">FRONT VIEW</span></div>
          <div className="body-stage">
            <div className="body-side body-side-right">R<span>PLAYER</span></div>
            <BodyMap key={mode} events={events} resetKey={inference ? runtime.resetKey : mockResetKey} selectedRegion={inference ? undefined : region} onSelectRegion={inference ? undefined : selectRegion} mediaClock={inference ? mediaClock : undefined} />
            <div className="body-side body-side-left">L<span>PLAYER</span></div>
            <div className="body-caption"><span className="status-dot neutral" />{bodyCaption}</div>
          </div>
          <div className="intensity-legend"><div><span>PERCEPTUAL INTENSITY</span><span>0.0 — 1.0</span></div><div className="legend-ramp" /><div className="legend-labels"><span>Low</span><span>Moderate</span><span>High</span></div></div>
          <p className="body-footnote">Anatomical left appears on your right.<br />Unknown regions leave the body map neutral.</p>
          <div className="representation-note"><span className="eyebrow">DEVICE-INDEPENDENT OUTPUT</span><p>Participant → body region → intensity</p><span>Intensity expresses perceptual salience.<br />It is not a measurement of physical force.</span></div>
        </aside>
      </section>
      <EventLog events={events} mock={!inference} />
      {inference && runtime.logPath && <p className="log-path" title={runtime.logPath}>Full model event stream saved by runtime: <code>{runtime.logPath}</code></p>}
      <footer className="page-footer"><span>UNIVERSAL HAPTIC MODEL <span className="footer-slash">/</span> V0 RESEARCH BASELINE</span><span>Physical meaning, before device output.</span></footer>
    </main>
  </div>
}
