export type ContactState = 'positive' | 'negative' | 'unknown' | 'unanalysed'
export interface ContactLabel {
  state: ContactState
  probability: number | null
  evidence_times_s?: number[]
  abstention_reason?: string
  basis?: 'visible' | 'inferred' | 'unknown'
}
export interface MotionEstimate {
  phase: string; feet: string; basis: 'visible' | 'inferred' | 'unknown'; cues: string[]
  evidence_times_s?: number[]; abstention_reason?: string
}
export interface ContactWindow {
  start_s: number; end_s: number; status: string; latency_s?: number
  labels: Record<string, ContactLabel>
  motion?: MotionEstimate
}
export interface ContactEvent {
  event_type: string; contact_region: string; start_s: number; end_s: number
  intensity: number | null; probability: number | null
  basis?: 'visible' | 'inferred' | 'unknown'
}
export interface ContactReport {
  duration_s: number; mode: string; task?: string; model: string; step_s: number
  actions: string[]; windows: ContactWindow[]; contact_events?: ContactEvent[]
  contact_regions?: Record<string, string>; evidence_policy?: string
  analysis_complete?: boolean
}
export const phaseNames: Record<string, string> = {
  running: '奔跑', walking: '行走', standing: '站立', takeoff: '起跳', airborne: '腾空',
  landing: '落地', cushioning: '缓冲', climbing: '攀爬 / 翻越', unknown: '待判断',
}
export const feetNames: Record<string, string> = {
  both: '双脚支撑', single: '单脚支撑', alternating: '交替支撑', none: '未支撑 · 腾空', unknown: '支撑方式待判断',
}
export const cueNames: Record<string, string> = {
  forward_flow: '场景向后流动', stride_bob: '步态周期起伏', launch: '起跳抬升', flight: '腾空轨迹',
  surface_approach: '接近落点', deceleration: '运动减速', compression_rebound: '压缩与回弹',
  visible_contact: '可见接触', hand_support: '手部支撑', camera_turn: '镜头转向', cut: '镜头切换', insufficient: '线索不足',
}
export const supportsContacts = (report: ContactReport) => ['visible_contact', 'egocentric_motion'].includes(report.task ?? '')
export function motionAt(report: ContactReport, seconds: number) {
  const window = windowAt(report, seconds)
  return report.mode !== 'dry_run' && window && validWindow(window) ? window.motion : undefined
}
export const contactNames: Record<string, string> = {
  foot_support: '足部支撑', foot_push_off: '蹬地起跳', foot_landing: '足部落地',
  hand_load: '手臂承载', hand_impact: '手部碰撞', arm_impact: '手臂碰撞',
  leg_impact: '腿部碰撞', torso_impact: '躯干着地 / 碰撞', head_impact: '头部碰撞',
  hip_support: '髋部支撑', torso_support: '躯干支撑',
  cushioning: '腿部缓冲',
}
export const contactRegions: Record<string, string> = {
  foot_support: 'feet', foot_push_off: 'feet', foot_landing: 'feet',
  hand_load: 'hands_and_arms', hand_impact: 'hands', arm_impact: 'arms',
  leg_impact: 'legs', torso_impact: 'torso', head_impact: 'head',
  hip_support: 'hips', torso_support: 'torso',
  cushioning: 'legs',
}
export const regionNames: Record<string, string> = {
  feet: '足部', hands: '手部', arms: '手臂', hands_and_arms: '手 / 臂',
  legs: '腿部', torso: '躯干', head: '头部', hips: '髋部', unknown: '未知区域',
}
export const validWindow = (window: ContactWindow) => ['ok', 'partial'].includes(window.status)

export function windowAt(report: ContactReport, seconds: number) {
  if (seconds < 0 || seconds >= report.duration_s || !Number.isFinite(seconds)) return undefined
  return report.windows.find(window => window.start_s <= seconds && seconds < window.end_s)
}

export function contactsAt(report: ContactReport, seconds: number): ContactEvent[] {
  const window = windowAt(report, seconds)
  if (report.mode === 'dry_run' || !supportsContacts(report) || !window || !validWindow(window)) return []
  return (report.contact_events ?? []).filter(event =>
    event.start_s <= seconds && seconds < event.end_s && window.labels[event.event_type]?.state === 'positive')
}

export function regionState(report: ContactReport, region: string, seconds: number): ContactState {
  const window = windowAt(report, seconds)
  if (!window || report.mode === 'dry_run' || !supportsContacts(report)) return 'unanalysed'
  if (!validWindow(window)) return 'unknown'
  const matches = (value: string) => value === region || (value === 'hands_and_arms' && ['hands', 'arms'].includes(region))
  const actions = report.actions.filter(action => matches((report.contact_regions ?? contactRegions)[action]))
  if (!actions.length) return 'unanalysed'
  if (contactsAt(report, seconds).some(event => matches(event.contact_region))) return 'positive'
  // A positive label without a matching candidate must not become a negative claim.
  return actions.every(action => window.labels[action]?.state === 'negative') ? 'negative' : 'unknown'
}

export const timecode = (seconds: number) => {
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`
}
