// A MediaPipe `.task` file is a plain zip archive of .tflite models (and nested
// .task archives), stored without compression. So reading one needs no zip
// library: walk the local file headers and slice each entry out of the buffer.
//
//   gesture_recognizer.task
//   ├── hand_landmarker.task
//   │   ├── hand_detector.tflite            palm detector, 192×192 input
//   │   └── hand_landmarks_detector.tflite  21 hand landmarks, 224×224 input
//   └── hand_gesture_recognizer.task
//       ├── gesture_embedder.tflite         landmarks → 128-d embedding
//       └── canned_gesture_classifier.tflite  embedding → 8 gestures

export async function loadTaskBundle(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`)
  return unzipStored(new Uint8Array(await response.arrayBuffer()))
}

// Returns { 'hand_detector.tflite': Uint8Array, ... }, flattening nested .task
// files. Entries are found through the zip's central directory (the index at
// the end of the file), because MediaPipe pads entries for alignment.
function unzipStored(bytes, files = {}) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let end = bytes.length - 22
  while (end >= 0 && view.getUint32(end, true) !== 0x06054b50) end-- // end-of-central-directory record
  if (end < 0) throw new Error('not a zip archive')
  const count = view.getUint16(end + 10, true)
  let p = view.getUint32(end + 16, true) // first central directory entry
  for (let i = 0; i < count; i++) {
    const method = view.getUint16(p + 10, true)
    const size = view.getUint32(p + 20, true)
    const nameLength = view.getUint16(p + 28, true)
    const extraLength = view.getUint16(p + 30, true)
    const commentLength = view.getUint16(p + 32, true)
    const local = view.getUint32(p + 42, true) // offset of the entry's local header
    const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nameLength))
    if (method !== 0) throw new Error(`${name}: only stored (uncompressed) zip entries are supported`)
    const start = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true)
    const data = bytes.subarray(start, start + size)
    if (name.endsWith('.task')) unzipStored(data, files)
    else files[name] = data
    p += 46 + nameLength + extraLength + commentLength
  }
  return files
}
