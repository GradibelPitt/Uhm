/* Generated from schemas/body_force_event.schema.json. Do not edit by hand. */

export type EventId = string
/**
 * Source video presentation time, not wall clock
 */
export type TimestampMs = number
export type PhysicalEventClass = 'impact' | 'collision' | 'explosion'
export type Recipient = 'player' | 'other_character' | 'environment' | 'unknown'
export type BodyRegion =
  | 'head'
  | 'left_torso'
  | 'right_torso'
  | 'lower_torso'
  | 'left_arm'
  | 'right_arm'
  | 'left_leg'
  | 'right_leg'
  | 'unknown'
  | 'whole_body'
/**
 * Perceptual salience, not physical force
 */
export type Intensity = number
export type DurationMs = number
export type Direction =
  'unknown' | 'front' | 'back' | 'left' | 'right' | 'front_left' | 'front_right' | 'above' | 'below'
export type Event = number
export type Recipient1 = number
export type Region = number
export type Intensity1 = number

export interface BodyForceEvent {
  event_id: EventId
  timestamp_ms: TimestampMs
  event_class: PhysicalEventClass
  recipient: Recipient
  body_region: BodyRegion
  intensity: Intensity
  duration_ms: DurationMs
  direction?: Direction
  confidence: Confidence
}
export interface Confidence {
  event: Event
  recipient: Recipient1
  region: Region
  intensity: Intensity1
}
