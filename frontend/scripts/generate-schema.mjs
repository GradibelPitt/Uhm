import { compileFromFile } from 'json-schema-to-typescript'
import { mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const schema = fileURLToPath(new URL('../../schemas/body_force_event.schema.json', import.meta.url))
const output = new URL('../src/generated/body-force-event.ts', import.meta.url)
await mkdir(new URL('../src/generated/', import.meta.url), { recursive: true })
await writeFile(output, await compileFromFile(schema, {
  bannerComment: '/* Generated from schemas/body_force_event.schema.json. Do not edit by hand. */',
  style: { singleQuote: true, semi: false },
}))
