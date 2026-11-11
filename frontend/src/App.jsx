import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, detectMode } from './api.js'
import MapView from './MapView.jsx'
import { Detail, Filters, Milestones, Trips, UploadDialog } from './Panels.jsx'

const bboxOf = (b) => b && [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(5)).join(',')

export default function App() {
  const [ready, setReady] = useState(false)
  const [staticMode, setStaticMode] = useState(false)
  const [filters, setFilters] = useState({ severity: '', trip: '' })
  const [heat, setHeat] = useState(false)
  const [areaOnly, setAreaOnly] = useState(false)
  const [bounds, setBounds] = useState(null)
  const [potholes, setPotholes] = useState([])
  const [allCounts, setAllCounts] = useState(null)
  const [stats, setStats] = useState(null)
  const [routes, setRoutes] = useState(null)
  const [trips, setTrips] = useState([])
  const [selected, setSelected] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [fitKey, setFitKey] = useState(0)
  const doneCount = useRef(0)

  useEffect(() => { detectMode().then((m) => { setStaticMode(m === 'static'); setReady(true) }) }, [])

  const query = useMemo(() => ({ ...filters, bbox: areaOnly ? bboxOf(bounds) : '' }), [filters, areaOnly, bounds])

  const load = useCallback(async () => {
    try {
      const mapFilters = { ...filters } // the map always shows everything that matches; "on screen" affects numbers
      const [ps, st, rt, all] = await Promise.all([api.potholes(mapFilters), api.stats(query), api.routes(filters),
        api.stats({ trip: filters.trip, bbox: query.bbox })])
      setPotholes(ps); setStats(st); setRoutes(rt); setAllCounts(all.by_severity); setError('')
    } catch (e) { setError(e.message) }
  }, [filters, query])

  const loadTrips = useCallback(async () => {
    try {
      const t = await api.trips()
      setTrips(t)
      const done = t.filter((x) => x.status === 'done').length
      if (done !== doneCount.current) { doneCount.current = done; setFitKey((k) => k + 1); load() }
      return t
    } catch (e) { setError(e.message); return [] }
  }, [load])

  useEffect(() => { if (ready) { load(); loadTrips() } }, [ready]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (ready) load() }, [query]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (ready) setFitKey((k) => k + 1) }, [filters.trip, filters.severity, ready])

  // poll while something is processing
  useEffect(() => {
    if (staticMode || !trips.some((t) => t.status === 'running' || t.status === 'queued')) return
    const h = setInterval(loadTrips, 1500)
    return () => clearInterval(h)
  }, [trips, staticMode, loadTrips])

  const runDemo = async () => {
    setBusy(true)
    try { await api.demo(); await loadTrips() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const remove = async (t) => {
    if (!window.confirm(`Delete "${t.name}" and the potholes only it found?`)) return
    try { await api.removeTrip(t.id); setSelected(null); doneCount.current = -1; await loadTrips(); await load() }
    catch (e) { setError(e.message) }
  }

  const processing = trips.find((t) => t.status === 'running' || t.status === 'queued')

  return (
    <div className={`app ${selected ? 'has-detail' : ''}`}>
      <aside className="side">
        <header className="brand">
          <svg viewBox="0 0 64 64" width="34" height="34" aria-hidden="true">
            <path d="M14 60V26a18 18 0 0 1 36 0v34z" fill="#fff" stroke="currentColor" strokeWidth="3" />
            <path d="M14 26a18 18 0 0 1 36 0z" fill="var(--cap)" stroke="currentColor" strokeWidth="3" />
            <ellipse cx="32" cy="45" rx="10" ry="5" fill="var(--large)" />
          </svg>
          <div>
            <h1>PotSpot</h1>
            <p>Potholes from your daily ride, one dot each</p>
          </div>
        </header>
        {staticMode && <p className="banner">Read-only demo: a simulated ride over held-out test photos. Run PotSpot on your computer to map your own roads.</p>}
        <Milestones stats={stats} />
        <Filters filters={filters} setFilters={setFilters} heat={heat} setHeat={setHeat}
          areaOnly={areaOnly} setAreaOnly={setAreaOnly} trips={trips} counts={allCounts} />
        <Trips trips={trips} onDemo={runDemo} onUpload={() => setUploading(true)} onDelete={remove}
          staticMode={staticMode} busy={busy || !!processing} />
        <section className="block">
          <a className={`btn wide ${stats?.potholes ? '' : 'is-disabled'}`} href={stats?.potholes ? api.reportUrl(query) : undefined}
            aria-disabled={!stats?.potholes}>Download report for the city (PDF)</a>
          <p className="fine">Worst potholes grouped by road, with photos and map links. Uses the filters above.</p>
        </section>
        {error && <p className="error" role="alert">{error}</p>}
      </aside>

      <main className="stage">
        <MapView potholes={potholes} routes={routes} heat={heat} selectedId={selected} onSelect={setSelected}
          onBounds={setBounds} fitKey={fitKey} />
        <div className="legend" aria-hidden="true">
          <span><i className="dot dot-large" />Large</span>
          <span><i className="dot dot-medium" />Medium</span>
          <span><i className="dot dot-small" />Small</span>
          <span><i className="route" />Route ridden</span>
        </div>
        {processing && (
          <div className="toast" role="status">
            <b>{processing.name}</b> {processing.message} ({Math.round(processing.progress * 100)}%)
          </div>
        )}
        {ready && !potholes.length && !processing && trips.length === 0 && (
          <div className="map-empty">
            <p>The map fills in as you add rides.</p>
          </div>
        )}
      </main>

      {selected && <Detail id={selected} onClose={() => setSelected(null)} />}
      {uploading && <UploadDialog onClose={() => setUploading(false)}
        onDone={() => { setUploading(false); loadTrips() }} />}
    </div>
  )
}
