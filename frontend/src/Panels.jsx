import { useEffect, useRef, useState } from 'react'
import { api, fmtDate, mapsLink } from './api.js'

const LABEL = { small: 'Small', medium: 'Medium', large: 'Large' }

// The signature element: summary numbers on Indian highway milestones.
export function Milestones({ stats }) {
  const items = [
    ['Potholes', stats?.potholes ?? '—', 'on the map'],
    ['Per km', stats?.per_km ?? '—', `${stats?.km ?? 0} km ridden`],
    ['Large', stats?.by_severity?.large ?? '—', 'need a fix first'],
    ['Repeat', stats?.seen_on_multiple_trips ?? '—', 'seen on 2+ rides'],
  ]
  return (
    <div className="milestones" aria-label="Summary">
      {items.map(([k, v, sub]) => (
        <div className="stone" key={k}>
          <div className="stone-cap">{k}</div>
          <div className="stone-body"><b>{v}</b><span>{sub}</span></div>
        </div>
      ))}
    </div>
  )
}

export function Filters({ filters, setFilters, heat, setHeat, areaOnly, setAreaOnly, trips, counts }) {
  const sev = filters.severity ? filters.severity.split(',') : []
  const toggle = (s) => {
    const next = sev.includes(s) ? sev.filter((x) => x !== s) : [...sev, s]
    setFilters({ ...filters, severity: next.length === 3 ? '' : next.join(',') })
  }
  return (
    <section className="block" aria-label="Filters">
      <h2>Show</h2>
      <div className="sev-chips" role="group" aria-label="Severity">
        {['large', 'medium', 'small'].map((s) => {
          const on = !sev.length || sev.includes(s)
          return (
            <button key={s} className={`chip chip-${s} ${on ? 'is-on' : ''}`} aria-pressed={on} onClick={() => toggle(s)}>
              <i aria-hidden="true" />{LABEL[s]} <span className="n">{counts?.[s] ?? 0}</span>
            </button>
          )
        })}
      </div>
      <label className="field">Ride
        <select value={filters.trip} onChange={(e) => setFilters({ ...filters, trip: e.target.value })}>
          <option value="">All rides</option>
          {trips.filter((t) => t.status === 'done').map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </label>
      <div className="toggles">
        <label className="switch"><input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /> Heatmap</label>
        <label className="switch"><input type="checkbox" checked={areaOnly} onChange={(e) => setAreaOnly(e.target.checked)} /> Only what's on screen</label>
      </div>
    </section>
  )
}

export function Trips({ trips, onDemo, onUpload, onDelete, staticMode, busy }) {
  return (
    <section className="block" aria-label="Rides">
      <div className="block-head">
        <h2>Rides</h2>
        {!staticMode && <button className="btn btn-primary" onClick={onUpload}>Add a ride</button>}
      </div>
      {trips.length === 0 && (
        <div className="empty">
          <p>No rides yet. Add a road video with its GPX track, or run the demo ride to see how it works.</p>
          {!staticMode && <button className="btn" onClick={onDemo} disabled={busy}>Run the demo ride</button>}
        </div>
      )}
      <ul className="trips">
        {trips.map((t) => (
          <li key={t.id} className={`trip trip-${t.status}`}>
            <div className="trip-top">
              <span className="trip-name">{t.name}</span>
              {!staticMode && t.status !== 'running' && t.status !== 'queued' &&
                <button className="link" onClick={() => onDelete(t)} aria-label={`Delete ${t.name}`}>Delete</button>}
            </div>
            {(t.status === 'running' || t.status === 'queued') && (
              <div className="progress" role="progressbar" aria-valuenow={Math.round(t.progress * 100)} aria-valuemin="0" aria-valuemax="100">
                <i style={{ width: `${Math.max(3, t.progress * 100)}%` }} />
              </div>
            )}
            <p className="trip-msg">
              {t.status === 'done'
                ? `${t.potholes_found} potholes · ${t.distance_km ?? 0} km · clocks: ${t.offset_method}${t.offset_s ? ` (${t.offset_s > 0 ? '+' : ''}${t.offset_s}s)` : ''}`
                : t.message}
            </p>
          </li>
        ))}
      </ul>
      {!staticMode && trips.length > 0 && !trips.some((t) => t.is_demo) &&
        <button className="link" onClick={onDemo} disabled={busy}>Run the demo ride</button>}
    </section>
  )
}

export function Detail({ id, onClose }) {
  const [p, setP] = useState(null)
  const [err, setErr] = useState('')
  const [copied, setCopied] = useState(false)
  const head = useRef(null)
  useEffect(() => {
    setP(null); setErr('')
    api.pothole(id).then(setP).catch((e) => setErr(e.message))
  }, [id])
  useEffect(() => { if (p) head.current?.focus() }, [p])
  const copy = async () => {
    try { await navigator.clipboard.writeText(`${p.lat.toFixed(6)}, ${p.lon.toFixed(6)}`); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* clipboard blocked */ }
  }
  return (
    <aside className="detail" aria-label="Pothole details">
      <button className="close" onClick={onClose} aria-label="Close details">×</button>
      {err && <p className="error">{err}</p>}
      {!p && !err && <p className="muted pad">Loading…</p>}
      {p && (
        <>
          <div className="photo">
            {p.image ? <img src={api.media(p.image)} alt={`Best photo of pothole ${p.id}`} /> : <span>No photo</span>}
            <span className={`badge badge-${p.severity}`}>{LABEL[p.severity]}</span>
          </div>
          <div className="pad">
            <h2 ref={head} tabIndex="-1">{p.road || 'Unnamed road'}</h2>
            <dl className="facts">
              <div><dt>Width (estimate)</dt><dd>{p.width_m ? `≈ ${p.width_m.toFixed(2)} m` : 'unknown'}</dd></div>
              <div><dt>Model confidence</dt><dd>{Math.round(p.confidence * 100)}%</dd></div>
              <div><dt>Seen</dt><dd>{p.sightings}× on {p.trips} ride{p.trips === 1 ? '' : 's'}</dd></div>
              <div><dt>Last seen</dt><dd>{fmtDate(p.last_seen)}</dd></div>
            </dl>
            <p className="coords"><span>{p.lat.toFixed(6)}, {p.lon.toFixed(6)}</span>
              <button className="link" onClick={copy}>{copied ? 'Copied' : 'Copy'}</button></p>
            <a className="btn btn-primary wide" href={mapsLink(p.lat, p.lon)} target="_blank" rel="noopener">Open in Google Maps</a>
            <h3>Sightings</h3>
            <ol className="sightings">
              {p.sightings_list.map((s, i) => (
                <li key={i}>
                  <img src={api.media(s.image)} alt="" loading="lazy" />
                  <div>
                    <b>{s.trip_name}</b>
                    <span>{fmtDate(s.seen_at)} · {LABEL[s.severity]}{s.width_m ? ` · ≈${s.width_m.toFixed(2)} m` : ''}</span>
                    <span>{s.frames} frames at {s.speed_kmh ?? '?'} km/h · {Math.round(s.confidence * 100)}%</span>
                  </div>
                </li>
              ))}
            </ol>
            <p className="note">Width comes from the pothole's size on screen and where it sits below the horizon,
              assuming a flat road. Depth cannot be seen by one camera, so check on site before ranking repairs.</p>
          </div>
        </>
      )}
    </aside>
  )
}

export function UploadDialog({ onClose, onDone }) {
  const [video, setVideo] = useState(null)
  const [gpx, setGpx] = useState(null)
  const [name, setName] = useState('')
  const [mode, setMode] = useState('auto')
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(false)
  const [pct, setPct] = useState(0)
  const [err, setErr] = useState('')
  const dlg = useRef(null)
  useEffect(() => { dlg.current?.showModal() }, [])

  const submit = (e) => {
    e.preventDefault()
    if (!video || !gpx) { setErr('Choose both the video and the GPX file from the same ride.'); return }
    const f = new FormData()
    f.append('video', video); f.append('gpx', gpx); f.append('name', name)
    f.append('offset_mode', mode); f.append('offset_s', String(offset))
    // XHR instead of fetch, for upload progress on big videos
    const x = new XMLHttpRequest()
    x.open('POST', 'api/trips')
    x.upload.onprogress = (ev) => ev.lengthComputable && setPct(ev.loaded / ev.total)
    x.onload = () => {
      setBusy(false)
      if (x.status === 202) { onDone(JSON.parse(x.responseText)); return }
      try { setErr(JSON.parse(x.responseText).detail) } catch { setErr(`Upload failed (${x.status}).`) }
    }
    x.onerror = () => { setBusy(false); setErr('Upload failed. Is PotSpot still running?') }
    setBusy(true); setErr(''); x.send(f)
  }

  return (
    <dialog ref={dlg} className="dialog" onClose={onClose} onCancel={onClose}>
      <form onSubmit={submit}>
        <h2>Add a ride</h2>
        <p className="muted">Record the road with your phone mounted facing forward, and a GPS logger app running on the same trip.</p>
        <label className="file">Road video (.mp4, .mov)
          <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files[0])} />
        </label>
        <label className="file">GPS track (.gpx)
          <input type="file" accept=".gpx,application/gpx+xml" onChange={(e) => setGpx(e.target.files[0])} />
        </label>
        <label className="field">Name (optional)
          <input type="text" value={name} placeholder="e.g. Kadri to Bejai, evening" onChange={(e) => setName(e.target.value)} />
        </label>
        <fieldset className="sync">
          <legend>Line up the clocks</legend>
          <label><input type="radio" name="m" checked={mode === 'auto'} onChange={() => setMode('auto')} />
            Automatic: match the video's movement to the GPS speed</label>
          <label><input type="radio" name="m" checked={mode === 'manual'} onChange={() => setMode('manual')} />
            I know it: the video started
            <input className="secs" type="number" step="0.5" value={offset} onChange={(e) => setOffset(e.target.value)}
              disabled={mode !== 'manual'} aria-label="seconds" /> seconds after the GPS app</label>
        </fieldset>
        {err && <p className="error" role="alert">{err}</p>}
        {busy && <div className="progress"><i style={{ width: `${Math.max(3, pct * 100)}%` }} /></div>}
        <div className="dialog-actions">
          <button type="button" className="btn" onClick={onClose} disabled={busy}>Cancel</button>
          <button type="submit" className="btn btn-primary" disabled={busy}>{busy ? `Uploading ${Math.round(pct * 100)}%` : 'Upload and find potholes'}</button>
        </div>
      </form>
    </dialog>
  )
}
