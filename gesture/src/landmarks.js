// Stage 4: the landmark model's output -> 21 points on the frame.
//
// The model sees only the 224×224 rotated crop, so its landmarks are in crop
// pixels. Mapping back is the inverse of the crop: normalise, un-rotate about
// the centre, scale to the rect, add the rect's position.
// (tensors_to_landmarks_calculator.cc, landmark_projection_calculator.cc)
//
// Outputs of hand_landmarks_detector.tflite, in order:
//   [63] landmarks: 21 × (x, y, z) in crop pixels; z is relative depth (wrist ≈ 0)
//   [1]  hand presence: probability that the crop contains a hand
//   [1]  handedness: probability of a *right* hand, assuming a mirrored (selfie) image
//   [63] world landmarks: 21 × (x, y, z) in metres, origin near the hand's centre

export const INPUT_SIZE = 224
const Z_SCALE = 0.4 // MediaPipe's normalize_z for this model

// The 21 points: wrist, then 4 joints per finger from the palm outwards.
export const LANDMARK_NAMES = [
  'wrist',
  'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
  'index_mcp', 'index_pip', 'index_dip', 'index_tip',
  'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
  'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
  'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

// Lines to draw between landmarks.
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]

// Crop pixels -> frame-normalised { x, y, z }. z uses the same scale as x.
export function projectLandmarks(raw, rect) {
  const cos = Math.cos(rect.rotation), sin = Math.sin(rect.rotation)
  const landmarks = []
  for (let i = 0; i < 21; i++) {
    const x = raw[3 * i] / INPUT_SIZE - 0.5
    const y = raw[3 * i + 1] / INPUT_SIZE - 0.5
    const z = raw[3 * i + 2] / INPUT_SIZE / Z_SCALE
    landmarks.push({
      x: (cos * x - sin * y) * rect.width + rect.xCenter,
      y: (sin * x + cos * y) * rect.height + rect.yCenter,
      z: z * rect.width,
    })
  }
  return landmarks
}

// World landmarks only need the crop's rotation undone.
// (world_landmark_projection_calculator.cc)
export function projectWorldLandmarks(raw, rect) {
  const cos = Math.cos(rect.rotation), sin = Math.sin(rect.rotation)
  const landmarks = []
  for (let i = 0; i < 21; i++) {
    const x = raw[3 * i], y = raw[3 * i + 1]
    landmarks.push({ x: cos * x - sin * y, y: sin * x + cos * y, z: raw[3 * i + 2] })
  }
  return landmarks
}
