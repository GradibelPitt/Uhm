import Ajv from 'ajv'
import schema from '../../schemas/body_force_event.schema.json'
import type { BodyForceEvent } from './generated/body-force-event'

export type { BodyForceEvent, BodyRegion } from './generated/body-force-event'

const ajv = new Ajv({ allErrors: true })
/** Inference and mock events are checked against the same exported Python schema. */
export const isBodyForceEvent = ajv.compile<BodyForceEvent>(schema)
