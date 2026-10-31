// WebGPU in Node: Dawn (Chrome's WebGPU implementation) as a native addon.
//
// `globals` holds the constructors and enums a browser puts on window
// (GPUBufferUsage, GPUTextureUsage, GPUMapMode, ...). Installing them on
// globalThis is what lets the *browser* shader code in ../segmentation run
// here unmodified.
import { create, globals } from 'webgpu'

Object.assign(globalThis, globals)

// Keep the GPU object referenced only from here: while it's reachable, Dawn
// keeps Node's event loop alive (see the webgpu package README, "Lifetime").
let gpu = create([])

export async function getDevice() {
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('No WebGPU adapter (is there a GPU / Vulkan driver?)')
  const device = await adapter.requestDevice()
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
  const info = adapter.info ?? {}
  return { device, description: [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') }
}

export function releaseGPU(device) {
  device.destroy()
  gpu = null
}
