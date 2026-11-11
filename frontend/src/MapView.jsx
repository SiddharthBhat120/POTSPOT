import { useEffect, useRef } from 'react'
import L from './leaflet-global.js'
import 'leaflet.markercluster'
import 'leaflet.heat'

const COLOR = { small: '#E2B400', medium: '#F07A1A', large: '#D7262E' }
const RANK = { small: 0, medium: 1, large: 2 }
const HEAT_WEIGHT = { small: 0.4, medium: 0.7, large: 1 }

function dotIcon(p, selected) {
  const size = 14 + RANK[p.severity] * 4 + (selected ? 6 : 0)
  return L.divIcon({
    className: '',
    html: `<span class="dot dot-${p.severity}${selected ? ' is-selected' : ''}" style="width:${size}px;height:${size}px"></span>`,
    iconSize: [size, size], iconAnchor: [size / 2, size / 2],
  })
}

function clusterIcon(cluster) {
  const kids = cluster.getAllChildMarkers()
  const worst = kids.reduce((w, m) => (RANK[m.options.sev] > RANK[w] ? m.options.sev : w), 'small')
  const n = kids.length
  const size = n < 10 ? 34 : n < 50 ? 42 : 50
  return L.divIcon({
    className: '',
    html: `<span class="cluster cluster-${worst}" style="width:${size}px;height:${size}px">${n}</span>`,
    iconSize: [size, size],
  })
}

// Plain Leaflet inside React: the map is created once; data layers are swapped when props change.
export default function MapView({ potholes, routes, heat, selectedId, onSelect, onBounds, fitKey }) {
  const box = useRef(null)
  const map = useRef(null)
  const layers = useRef({})
  const markers = useRef(new Map())
  const fitted = useRef(null)

  useEffect(() => {
    const m = L.map(box.current, { zoomControl: false, preferCanvas: true }).setView([12.87, 74.84], 13)
    L.control.zoom({ position: 'bottomright' }).addTo(m)
    L.control.scale({ position: 'bottomleft', imperial: false }).addTo(m)
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(m)
    layers.current.routes = L.layerGroup().addTo(m)
    layers.current.cluster = L.markerClusterGroup({
      iconCreateFunction: clusterIcon, showCoverageOnHover: false, maxClusterRadius: 40, spiderfyOnMaxZoom: true,
    }).addTo(m)
    const report = () => onBounds?.(m.getBounds())
    m.on('moveend', report)
    map.current = m
    return () => m.remove()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // routes
  useEffect(() => {
    const g = layers.current.routes
    g.clearLayers()
    for (const f of routes?.features || []) {
      const ll = f.geometry.coordinates.map(([lon, lat]) => [lat, lon])
      L.polyline(ll, { color: '#FFFFFF', weight: 9, opacity: 0.9 }).addTo(g)
      L.polyline(ll, { color: '#0B6E4F', weight: 4, opacity: 0.95, dashArray: f.properties.demo ? '10 7' : null })
        .bindTooltip(f.properties.name, { sticky: true }).addTo(g)
    }
  }, [routes])

  // markers and heat
  useEffect(() => {
    const m = map.current
    const c = layers.current.cluster
    c.clearLayers()
    markers.current.clear()
    if (layers.current.heat) { m.removeLayer(layers.current.heat); layers.current.heat = null }
    if (heat) {
      layers.current.heat = L.heatLayer(potholes.map((p) => [p.lat, p.lon, HEAT_WEIGHT[p.severity]]),
        { radius: 26, blur: 20, maxZoom: 17, minOpacity: 0.35,
          gradient: { 0.3: '#E2B400', 0.6: '#F07A1A', 1: '#D7262E' } }).addTo(m)
    } else {
      for (const p of potholes) {
        const mk = L.marker([p.lat, p.lon], { icon: dotIcon(p, p.id === selectedId), sev: p.severity, keyboard: true,
          title: `${p.severity} pothole${p.road ? ` on ${p.road}` : ''}` })
        mk.on('click', () => onSelect(p.id))
        markers.current.set(p.id, mk)
        c.addLayer(mk)
      }
    }
    // fit once per data set (new filters or new trip), not on every refresh
    if (fitted.current !== fitKey) {
      const pts = [...potholes.map((p) => [p.lat, p.lon]),
        ...(routes?.features || []).flatMap((f) => f.geometry.coordinates.map(([lon, lat]) => [lat, lon]))]
      if (pts.length) { m.fitBounds(L.latLngBounds(pts).pad(0.15), { maxZoom: 17 }); fitted.current = fitKey }
    }
  }, [potholes, heat, fitKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // selection: restyle the chosen dot and bring it into view
  useEffect(() => {
    for (const [id, mk] of markers.current) {
      const p = potholes.find((x) => x.id === id)
      if (p) mk.setIcon(dotIcon(p, id === selectedId))
    }
    const mk = markers.current.get(selectedId)
    if (mk) layers.current.cluster.zoomToShowLayer(mk, () => map.current.panTo(mk.getLatLng(), { animate: true }))
  }, [selectedId]) // eslint-disable-line react-hooks/exhaustive-deps

  return <div className="map" ref={box} role="application" aria-label="Map of potholes" />
}
