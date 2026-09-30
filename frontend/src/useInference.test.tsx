import { act, cleanup, renderHook } from '@testing-library/react'
import type { RefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useInference } from './useInference'
import type { BodyForceEvent } from './event'

class TestSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3
  static instances: TestSocket[] = []
  readyState = TestSocket.CONNECTING
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  messages: Array<{ type: string; request_id: number; timestamp_ms?: number }> = []
  constructor(public url: string) { TestSocket.instances.push(this) }
  send(value: string) { this.messages.push(JSON.parse(value)) }
  close() { this.readyState = TestSocket.CLOSED; this.onclose?.() }
  emit(value: object) { this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent) }
}

const sampleEvent: BodyForceEvent = {
  event_id: 'real-model-event', timestamp_ms: 1000, event_class: 'impact', recipient: 'player',
  body_region: 'left_torso', intensity: 0.6, duration_ms: 200, direction: 'unknown',
  confidence: { event: 0.9, recipient: 0.9, region: 0.8, intensity: 0 },
}

async function connect() {
  const mediaState = { paused: false, currentTime: 1, duration: 20, seeking: false, ended: false, readyState: 4 }
  const media = document.createElement('video')
  for (const key of Object.keys(mediaState) as Array<keyof typeof mediaState>) {
    Object.defineProperty(media, key, { get: () => mediaState[key], configurable: true })
  }
  const video: RefObject<HTMLVideoElement | null> = { current: media }
  const hook = renderHook(() => useInference(video, true))
  await act(async () => {})
  const socket = TestSocket.instances.at(-1)!
  act(() => {
    socket.readyState = TestSocket.OPEN
    socket.emit({ type: 'ready', session_id: 'test-session', log_path: 'test.events.jsonl' })
  })
  const reset = socket.messages.at(-1)!
  act(() => socket.emit({ type: 'reset', request_id: reset.request_id }))
  return { ...hook, socket, mediaState }
}
const tick = (ms = 125) => act(() => vi.advanceTimersByTime(ms))

describe('inference transport scheduling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    TestSocket.instances = []
    vi.stubGlobal('WebSocket', TestSocket)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      video_url: '/api/video', video_name: 'demo.mp4', video: { duration_ms: 20000, fps: 12 },
      model: { name: 'checkpoint', device: 'cpu', checkpoint: 'baseline.pt', synthetic: true },
      schema_version: '0.1', sample_step_ms: 125,
    }) }))
    vi.spyOn(console, 'info').mockImplementation(() => {})
  })
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers() })

  it('allows only one outstanding sample and samples only while playback is advancing', async () => {
    const { socket, mediaState } = await connect()
    tick(1250)
    const samples = () => socket.messages.filter(message => message.type === 'sample')
    expect(samples()).toHaveLength(1)
    act(() => socket.emit({ type: 'prediction', request_id: samples()[0].request_id, event: null }))
    tick(500)
    expect(samples()).toHaveLength(1)
    mediaState.currentTime = 2
    mediaState.paused = true
    tick(500)
    expect(samples()).toHaveLength(1)
    mediaState.paused = false
    tick()
    expect(samples()).toHaveLength(2)
  })

  it('ignores stale replies after a seek and waits for the matching reset acknowledgment', async () => {
    const { socket, result, mediaState } = await connect()
    tick()
    const oldRequest = socket.messages.at(-1)!
    act(() => result.current.reset())
    const newReset = socket.messages.at(-1)!
    act(() => socket.emit({ type: 'prediction', request_id: oldRequest.request_id, event: sampleEvent, diagnostics: { event_class: 'impact' } }))
    expect(result.current.events).toHaveLength(0)
    expect(result.current.diagnostics).toBeNull()
    act(() => socket.emit({ type: 'reset', request_id: 0 }))
    tick(500)
    expect(socket.messages.at(-1)?.type).toBe('reset')
    act(() => socket.emit({ type: 'reset', request_id: newReset.request_id }))
    mediaState.currentTime = 3
    tick()
    const newRequest = socket.messages.at(-1)!
    expect(newRequest.type).toBe('sample')
    expect(newRequest.request_id).toBeGreaterThan(newReset.request_id)
    act(() => socket.emit({ type: 'prediction', request_id: newRequest.request_id, event: sampleEvent }))
    expect(result.current.events).toEqual([sampleEvent])
  })

  it('keeps the final sample strictly below the video duration', async () => {
    const { socket, mediaState } = await connect()
    mediaState.currentTime = 19.999999
    tick()
    expect(socket.messages.at(-1)?.timestamp_ms).toBe(19999)
  })

  it('clears the latest response on seek while preserving the historical event log', async () => {
    const { socket, result, mediaState } = await connect()
    tick()
    act(() => socket.emit({ type: 'prediction', request_id: socket.messages.at(-1)!.request_id, event: sampleEvent }))
    expect(result.current.latestEvent).toEqual(sampleEvent)
    mediaState.currentTime = 0.5
    act(() => result.current.reset())
    expect(result.current.latestEvent).toBeNull()
    expect(result.current.events).toEqual([sampleEvent])
    const reset = socket.messages.at(-1)!
    act(() => socket.emit({ type: 'reset', request_id: reset.request_id }))
    tick()
    const nextEvent = { ...sampleEvent, event_id: 'replayed-head-impact', body_region: 'head', timestamp_ms: 500 }
    act(() => socket.emit({ type: 'prediction', request_id: socket.messages.at(-1)!.request_id, event: nextEvent }))
    expect(result.current.latestEvent).toEqual(nextEvent)
    expect(result.current.events).toEqual([sampleEvent, nextEvent])
  })

  it('clears the latest response when reconnecting without deleting prior event evidence', async () => {
    const { socket, result } = await connect()
    tick()
    act(() => socket.emit({ type: 'prediction', request_id: socket.messages.at(-1)!.request_id, event: sampleEvent }))
    expect(result.current.latestEvent).toEqual(sampleEvent)
    await act(async () => result.current.retry())
    expect(result.current.latestEvent).toBeNull()
    expect(result.current.events).toEqual([sampleEvent])
    const reconnected = TestSocket.instances.at(-1)!
    expect(reconnected).not.toBe(socket)
    act(() => {
      reconnected.readyState = TestSocket.OPEN
      reconnected.emit({ type: 'ready', session_id: 'reconnected' })
    })
    expect(result.current.latestEvent).toBeNull()
    expect(result.current.events).toEqual([sampleEvent])
  })

  it('rejects a server event that violates the canonical schema', async () => {
    const { socket, result } = await connect()
    tick()
    const request = socket.messages.at(-1)!
    act(() => socket.emit({ type: 'prediction', request_id: request.request_id, event: { ...sampleEvent, duration_ms: 0 } }))
    expect(result.current.events).toHaveLength(0)
    expect(result.current.connection).toBe('error')
    expect(result.current.error).toContain('canonical BodyForceEvent schema')
  })
})
