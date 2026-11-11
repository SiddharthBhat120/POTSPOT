import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'leaflet/dist/leaflet.css'
import 'leaflet.markercluster/dist/MarkerCluster.css'
import App from './App.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
