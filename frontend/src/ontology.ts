import { parse } from 'yaml'
import ontologySource from '../../ontology/body_regions.yaml?raw'
import type { BodyRegion } from './event'

export interface RegionGeometry { id: BodyRegion; label: string; path: string }
interface BodyOntology {
  version: string
  viewBox: string
  regions: RegionGeometry[]
  special_regions: BodyRegion[]
}

export const ontology = parse(ontologySource) as BodyOntology
export const isBodyRegion = (value: string): value is BodyRegion =>
  ontology.regions.some(region => region.id === value) || ontology.special_regions.some(region => region === value)
export const regionLabel = (id: string) => ontology.regions.find(region => region.id === id)?.label
  ?? id.replaceAll('_', ' ').replace(/^./, first => first.toUpperCase())
