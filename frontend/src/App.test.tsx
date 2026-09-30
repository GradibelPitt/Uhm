import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { useInference } from './useInference'
import type { RuntimeMetadata } from './useInference'

vi.mock('./useInference', () => ({ useInference: vi.fn() }))

function renderSource(viewpoint: RuntimeMetadata['video']['viewpoint'], synthetic = false) {
  vi.mocked(useInference).mockReturnValue({
    metadata: {
      video_url: '/api/video', video_name: 'demo.mp4',
      video: { fps: 12, duration_ms: 20000, synthetic, viewpoint },
      model: { name: 'Test model', checkpoint: 'baseline.pt', device: 'cpu', synthetic: true },
      schema_version: '0.1',
    },
    connection: 'ready', error: null, events: [], latestEvent: null,
    diagnostics: null, stats: {}, logPath: '', resetKey: 0, reset: vi.fn(), retry: vi.fn(),
  })
  return render(<App />)
}

describe('source video viewpoint disclosure', () => {
  afterEach(cleanup)

  it('labels the known third-person front-view demo while retaining its synthetic disclosure', () => {
    renderSource('third_person', true)
    expect(screen.getByText('Third-person · front view')).toBeTruthy()
    expect(screen.getByText('SYNTHETIC VIDEO — PIPELINE TEST')).toBeTruthy()
  })

  it.each(['unknown', undefined] as const)('does not infer a viewpoint from the model or filename when it is %s', viewpoint => {
    renderSource(viewpoint)
    expect(screen.getByText('Viewpoint unspecified')).toBeTruthy()
    expect(screen.queryByText('Third-person · front view')).toBeNull()
    expect(screen.queryByText('First-person')).toBeNull()
    expect(screen.queryByText('SYNTHETIC VIDEO — PIPELINE TEST')).toBeNull()
  })

  it('keeps explicitly declared first-person footage distinct from the third-person demo', () => {
    renderSource('first_person')
    expect(screen.getByText('First-person')).toBeTruthy()
    expect(screen.queryByText('Third-person · front view')).toBeNull()
  })
})
