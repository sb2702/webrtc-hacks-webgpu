// Turning detections and landmarks into the rotated square the landmark model
// looks at. A straight port of MediaPipe's calculators (file names below), so
// that our crops, and therefore our landmarks, match MediaPipe's.
//
// A rect is { xCenter, yCenter, width, height, rotation }: centre and size
// normalised to the frame (0..1), rotation in radians, clockwise on screen.
// Rotations are computed in pixels, because a normalised square isn't square
// unless the frame is.

const normalizeRadians = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI))

// Rotation that turns the vector from point (x0, y0) to (x1, y1) upright, i.e.
// so the hand points up in the crop. (detections_to_rects_calculator.cc)
function uprightRotation(x0, y0, x1, y1) {
  return normalizeRadians(Math.PI / 2 - Math.atan2(-(y1 - y0), x1 - x0))
}

// Grow / shift a rect and make it square in pixels. (rect_transformation_calculator.cc)
// The shift is along the rect's own (rotated) axes.
function transformRect(rect, frameWidth, frameHeight, { scale, shiftY }) {
  const { width, height, rotation } = rect
  const xShift = (-frameHeight * height * shiftY * Math.sin(rotation)) / frameWidth
  const yShift = (frameHeight * height * shiftY * Math.cos(rotation)) / frameHeight
  const longSide = Math.max(width * frameWidth, height * frameHeight)
  return {
    xCenter: rect.xCenter + xShift,
    yCenter: rect.yCenter + yShift,
    width: (longSide / frameWidth) * scale,
    height: (longSide / frameHeight) * scale,
    rotation,
  }
}

// Palm detection -> hand rect. The detector finds the palm; the hand extends
// well beyond it towards the fingers, so the box is scaled 2.6× and shifted
// half its height "up" the hand. Keypoint 0 is the wrist, keypoint 2 the base
// of the middle finger. (hand_detector_graph.cc)
export function palmToRect(palm, frameWidth, frameHeight) {
  const [xmin, ymin, xmax, ymax] = palm.box
  const k = palm.keypoints
  const rotation = uprightRotation(k[0] * frameWidth, k[1] * frameHeight, k[4] * frameWidth, k[5] * frameHeight)
  const rect = { xCenter: (xmin + xmax) / 2, yCenter: (ymin + ymax) / 2, width: xmax - xmin, height: ymax - ymin, rotation }
  return transformRect(rect, frameWidth, frameHeight, { scale: 2.6, shiftY: -0.5 })
}

// Landmarks from this frame -> where to crop in the next one (tracking).
// Upright direction: wrist -> the average of the index, middle and ring finger
// bases. Box: the landmarks' bounds measured along the rotated axes, then
// grown 2× so the hand stays inside if it moves. Like MediaPipe, only the
// palm and lower finger joints are used: fingertips move too much.
// (hand_landmarks_to_rect_calculator.cc, hand_landmarks_detector_graph.cc)
const TRACKING_LANDMARKS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

export function landmarksToRect(landmarks, frameWidth, frameHeight) {
  const px = (i) => landmarks[i].x * frameWidth
  const py = (i) => landmarks[i].y * frameHeight
  const x1 = ((px(5) + px(13)) / 2 + px(9)) / 2
  const y1 = ((py(5) + py(13)) / 2 + py(9)) / 2
  const rotation = uprightRotation(px(0), py(0), x1, y1)

  // Bounds in the rotated frame, around the axis-aligned centre.
  const xs = TRACKING_LANDMARKS.map(px), ys = TRACKING_LANDMARKS.map(py)
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2
  const cos = Math.cos(-rotation), sin = Math.sin(-rotation)
  const rx = xs.map((x, i) => (x - cx) * cos - (ys[i] - cy) * sin)
  const ry = xs.map((x, i) => (x - cx) * sin + (ys[i] - cy) * cos)
  const [minX, maxX, minY, maxY] = [Math.min(...rx), Math.max(...rx), Math.min(...ry), Math.max(...ry)]
  const pcx = (minX + maxX) / 2, pcy = (minY + maxY) / 2 // centre in the rotated frame, rotated back:
  const rect = {
    xCenter: (pcx * Math.cos(rotation) - pcy * Math.sin(rotation) + cx) / frameWidth,
    yCenter: (pcx * Math.sin(rotation) + pcy * Math.cos(rotation) + cy) / frameHeight,
    width: (maxX - minX) / frameWidth,
    height: (maxY - minY) / frameHeight,
    rotation,
  }
  return transformRect(rect, frameWidth, frameHeight, { scale: 2.0, shiftY: -0.1 })
}

// The four corners of a rect in frame-normalised coordinates, for drawing.
export function rectCorners(rect, frameWidth, frameHeight) {
  const cos = Math.cos(rect.rotation), sin = Math.sin(rect.rotation)
  const w = rect.width * frameWidth, h = rect.height * frameHeight
  return [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([u, v]) => ({
    x: rect.xCenter + (cos * u * w - sin * v * h) / frameWidth,
    y: rect.yCenter + (sin * u * w + cos * v * h) / frameHeight,
  }))
}
