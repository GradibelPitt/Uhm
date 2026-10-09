import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ContactWorkbench } from './ContactWorkbench'

const report = {
  duration_s: 2, mode: 'local_vlm', task: 'visible_contact', model: 'test-model', step_s: .5,
  actions: ['foot_support'],
  windows: [{ start_s: 0, end_s: .5, status: 'ok', labels: { foot_support: { state: 'positive', probability: null } } }],
  contact_events: [{ event_type: 'foot_support', contact_region: 'feet', start_s: 0, end_s: .5, intensity: null, probability: null }],
}
const meta = {
  video_url: '/api/video', video_name: 'jump.mp4', video: { fps: 25, width: 320, height: 240, duration_ms: 2000 },
  playback: { model: 'test-model', request_p50_s: 4, request_p95_s: 5, request_to_step_ratio: 8, analysis_wall_s: 16 },
}
afterEach(() => { cleanup(); vi.unstubAllGlobals() })
function serve(dryRun = false) {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => ({ ok: true, json: async () => url === '/api/meta' ? meta : { ...report, mode: dryRun ? 'dry_run' : report.mode } })))
}
describe('contact workbench', () => {
  it('synchronizes seek and video failure without stale highlights', async () => {
    serve()
    const view = render(<ContactWorkbench />)
    const media = await screen.findByLabelText('运动源视频') as HTMLVideoElement
    expect(screen.getByText('等待视频加载')).toBeTruthy()
    fireEvent.loadedData(media)
    expect(view.container.querySelector('[data-region="feet"]')?.getAttribute('class')).toContain('positive')
    media.currentTime = 1.1
    fireEvent.seeking(media)
    expect(view.container.querySelector('[data-region="feet"]')?.getAttribute('class')).toContain('unanalysed')
    media.currentTime = .2
    fireEvent.seeking(media)
    expect(view.container.querySelector('[data-region="feet"]')?.getAttribute('class')).toContain('positive')
    fireEvent.error(media)
    expect(screen.getByRole('alert')).toBeTruthy()
    expect(view.container.querySelectorAll('.contact-part.positive').length).toBe(0)
  })
  it('keeps dry runs neutral even if a report contains stale positive segments', async () => {
    serve(true)
    const view = render(<ContactWorkbench />)
    fireEvent.loadedData(await screen.findByLabelText('运动源视频'))
    expect(screen.getByText('尚未运行模型')).toBeTruthy()
    expect(view.container.querySelectorAll('.contact-part.positive').length).toBe(0)
    expect(view.container.querySelectorAll('.contact-segment').length).toBe(0)
  })
  it('offers retry after an API failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 503 })))
    render(<ContactWorkbench />)
    expect(await screen.findByText('工作台暂时无法连接')).toBeTruthy()
    serve()
    fireEvent.click(screen.getByRole('button', { name: '重新连接' }))
    await waitFor(() => expect(screen.getByLabelText('运动源视频')).toBeTruthy())
  })
})
