// Two modes: "live" talks to the FastAPI backend; "static" reads a snapshot of the demo ride
// (used for the read-only demo on GitHub Pages, where there is no server).
const SEV = ['small', 'medium', 'large']
let mode = null
let snap = null

async function req(path, opts = {}) {
  const res = await fetch(path, opts)
  if (!res.ok) {
    let msg = `The server answered ${res.status}.`
    try { const b = await res.json(); if (b.detail) msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail) } catch { /* empty */ }
    throw new Error(msg)
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res
}

export async function detectMode() {
  if (mode) return mode
  try {
    const r = await fetch('api/health')
    if (r.ok && r.headers.get('content-type')?.includes('json')) { mode = 'live'; return mode }
  } catch { /* no server */ }
  try {
    snap = await (await fetch('snapshot/snapshot.json')).json()
    mode = 'static'
  } catch {
    mode = 'live' // show live errors rather than nothing
  }
  return mode
}

const qs = (f) => new URLSearchParams(Object.entries(f).filter(([, v]) => v !== '' && v != null)).toString()

function filterStatic(f) {
  let ps = snap.potholes
  if (f.severity) ps = ps.filter((p) => f.severity.split(',').includes(p.severity))
  if (f.bbox) {
    const [w, s, e, n] = f.bbox.split(',').map(Number)
    ps = ps.filter((p) => p.lon >= w && p.lon <= e && p.lat >= s && p.lat <= n)
  }
  if (f.trip) ps = ps.filter((p) => snap.details[p.id].sightings_list.some((x) => x.trip_id === f.trip))
  return ps
}

export const api = {
  get mode() { return mode },
  health: () => mode === 'static' ? Promise.resolve({ ok: true, static: true }) : req('api/health'),
  potholes: (f) => mode === 'static' ? Promise.resolve(filterStatic(f)) : req(`api/potholes?${qs(f)}`),
  pothole: (id) => mode === 'static' ? Promise.resolve(snap.details[id]) : req(`api/potholes/${id}`),
  stats: (f) => {
    if (mode !== 'static') return req(`api/stats?${qs(f)}`)
    const ps = filterStatic(f)
    const km = snap.trips.filter((t) => !f.trip || t.id === f.trip).reduce((a, t) => a + (t.distance_km || 0), 0)
    const by = Object.fromEntries(SEV.map((s) => [s, ps.filter((p) => p.severity === s).length]))
    return Promise.resolve({ potholes: ps.length, by_severity: by, km: Math.round(km * 100) / 100,
      per_km: km ? Math.round(ps.length / km * 100) / 100 : null, trips: snap.trips.length,
      seen_on_multiple_trips: ps.filter((p) => p.trips > 1).length })
  },
  routes: (f) => mode === 'static' ? Promise.resolve(snap.routes) : req(`api/routes?${qs({ trip: f.trip })}`),
  trips: () => mode === 'static' ? Promise.resolve(snap.trips) : req('api/trips'),
  trip: (id) => req(`api/trips/${id}`),
  demo: () => req('api/demo', { method: 'POST' }),
  upload: (form) => req('api/trips', { method: 'POST', body: form }),
  removeTrip: (id) => req(`api/trips/${id}`, { method: 'DELETE' }),
  reportUrl: (f) => mode === 'static' ? 'snapshot/report.pdf' : `api/report.pdf?${qs(f)}`,
  media: (p) => p ? (mode === 'static' ? `snapshot/media/${p}` : `media/${p}`) : null,
}

export const SEVERITIES = SEV
export const fmtDate = (s) => s ? new Date(s).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
export const mapsLink = (lat, lon) => `https://www.google.com/maps/search/?api=1&query=${lat.toFixed(6)},${lon.toFixed(6)}`
