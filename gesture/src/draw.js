// Canvas 2D overlays: hand skeleton, crop rect, palm box. Coordinates are
// frame-normalised; the canvas is assumed to be the frame's size.
import { HAND_CONNECTIONS } from './landmarks.js'
import { rectCorners } from './rects.js'

export function drawHand(ctx, landmarks, { color = '#00e676', radius = 4 } = {}) {
  const { width, height } = ctx.canvas
  const scale = Math.max(1, width / 640)
  ctx.save()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 2 * scale
  ctx.beginPath()
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.moveTo(landmarks[a].x * width, landmarks[a].y * height)
    ctx.lineTo(landmarks[b].x * width, landmarks[b].y * height)
  }
  ctx.stroke()
  for (const p of landmarks) {
    ctx.beginPath()
    ctx.arc(p.x * width, p.y * height, radius * scale, 0, 2 * Math.PI)
    ctx.fill()
  }
  ctx.restore()
}

// The rotated square the landmark model was given.
export function drawRect(ctx, rect, { color = '#ffd600' } = {}) {
  const { width, height } = ctx.canvas
  const corners = rectCorners(rect, width, height)
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 2 * Math.max(1, width / 640)
  ctx.setLineDash([8, 6])
  ctx.beginPath()
  corners.forEach((c, i) => ctx[i ? 'lineTo' : 'moveTo'](c.x * width, c.y * height))
  ctx.closePath()
  ctx.stroke()
  // Mark the crop's "top" edge, where the fingers should point.
  ctx.setLineDash([])
  ctx.lineWidth *= 2
  ctx.beginPath()
  ctx.moveTo(corners[0].x * width, corners[0].y * height)
  ctx.lineTo(corners[1].x * width, corners[1].y * height)
  ctx.stroke()
  ctx.restore()
}

// The palm detector's box and its 7 keypoints (only on frames where it ran).
export function drawPalm(ctx, palm, { color = '#29b6f6' } = {}) {
  const { width, height } = ctx.canvas
  const [xmin, ymin, xmax, ymax] = palm.box
  ctx.save()
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = 2 * Math.max(1, width / 640)
  ctx.strokeRect(xmin * width, ymin * height, (xmax - xmin) * width, (ymax - ymin) * height)
  for (let k = 0; k < 7; k++) {
    ctx.beginPath()
    ctx.arc(palm.keypoints[2 * k] * width, palm.keypoints[2 * k + 1] * height, 3 * Math.max(1, width / 640), 0, 2 * Math.PI)
    ctx.fill()
  }
  ctx.restore()
}
