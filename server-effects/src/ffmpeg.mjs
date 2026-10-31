// ffmpeg as the decoder and encoder: raw RGBA frames over pipes.
//
// On a server there's no WebCodecs and no VideoFrame, so frames arrive and
// leave as bytes in CPU memory. ffmpeg accepts any input it understands (a
// file, RTMP, SRT, RTSP, ...) and writes any output.
import { spawn, execFileSync } from 'node:child_process'

export function probe(input) {
  const out = execFileSync('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', input,
  ])
  const { width, height, r_frame_rate } = JSON.parse(out).streams[0]
  const [num, den] = r_frame_rate.split('/').map(Number)
  return { width, height, fps: num / den, rate: r_frame_rate }
}

// Yields one Buffer of width × height × 4 bytes (RGBA) per frame.
export async function* decode(input, { width, height }) {
  const ffmpeg = spawn('ffmpeg', ['-v', 'error', '-i', input, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], { stdio: ['ignore', 'pipe', 'inherit'] })
  const frameBytes = width * height * 4
  let pending = []
  let pendingBytes = 0
  for await (const chunk of ffmpeg.stdout) {
    pending.push(chunk)
    pendingBytes += chunk.length
    while (pendingBytes >= frameBytes) {
      const all = Buffer.concat(pending)
      yield all.subarray(0, frameBytes)
      pending = [all.subarray(frameBytes)]
      pendingBytes -= frameBytes
    }
  }
}

export function encode(output, { width, height, rate }, encoder) {
  const args = ['-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${width}x${height}`, '-r', rate, '-i', '-',
    '-c:v', encoder, ...(encoder === 'libx264' ? ['-preset', 'veryfast'] : []), '-pix_fmt', 'yuv420p', output]
  const ffmpeg = spawn('ffmpeg', args, { stdio: ['pipe', 'ignore', 'inherit'] })
  const done = new Promise((resolve, reject) => ffmpeg.on('close', (code) => (code ? reject(new Error(`ffmpeg exited with ${code}`)) : resolve())))
  return {
    // Respect backpressure: wait when the pipe is full.
    write: (rgba) => (ffmpeg.stdin.write(rgba) ? Promise.resolve() : new Promise((r) => ffmpeg.stdin.once('drain', r))),
    close: () => (ffmpeg.stdin.end(), done),
  }
}
