import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { HapticPreviewPanel } from './HapticPreviewPanel'

const preview = { schema_version: '0.1', mode: 'design_preview',
  amplitude_semantics: 'unit_envelope_not_force', duration_ms: 1000, inferred_included: false,
  cues: [{ cue_id: 'cue-0', source_ids: ['contact-0'], event_type: 'foot_landing', region: 'feet',
    basis: 'visible', effect: 'pulse', start_ms: 100, end_ms: 320,
    envelope: [{ offset_ms: 0, level: 0 }, { offset_ms: 22, level: 1 }, { offset_ms: 220, level: 0 }] }],
}
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('haptic preview panel', () => {
  it('follows media time, seeks through the cue, and clears when video is unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => preview })))
    const onSeek = vi.fn()
    const view = render(<HapticPreviewPanel seconds={.122} onSeek={onSeek} />)
    const meter = await screen.findByLabelText('足部 · 脉冲当前包络') as HTMLMeterElement
    expect(meter.value).toBe(1)
    fireEvent.click(screen.getByRole('button', { name: /定位触觉/ }))
    expect(onSeek).toHaveBeenCalledWith(.1)
    view.rerender(<HapticPreviewPanel seconds={-1} onSeek={onSeek} />)
    expect(meter.value).toBe(0)
  })
  it('requires explicit opt-in for inferred cues and ignores a late old response', async () => {
    let resolveOld: (value: unknown) => void = () => {}
    const fetcher = vi.fn().mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
      .mockResolvedValue({ ok: true, json: async () => ({ ...preview, inferred_included: true, cues: [] }) })
    vi.stubGlobal('fetch', fetcher)
    render(<HapticPreviewPanel seconds={0} onSeek={() => {}} />)
    expect(fetcher.mock.calls[0][0]).toContain('include_inferred=false')
    fireEvent.click(screen.getByLabelText('纳入运动推断'))
    await screen.findByText(/当前没有符合条件的接触/)
    expect(fetcher.mock.calls[1][0]).toContain('include_inferred=true')
    resolveOld({ ok: true, json: async () => preview })
    await waitFor(() => expect(screen.queryByLabelText('足部 · 脉冲当前包络')).toBeNull())
  })
  it('keeps preview errors local and supports retry', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce({ ok: false })
      .mockResolvedValue({ ok: true, json: async () => preview })
    vi.stubGlobal('fetch', fetcher)
    render(<HapticPreviewPanel seconds={0} onSeek={() => {}} />)
    fireEvent.click(await screen.findByText('重试预览'))
    expect(await screen.findByLabelText('足部 · 脉冲触觉包络')).toBeTruthy()
  })
})
