// Leaflet plugins (markercluster, heat) look for a global `L`. This module runs before them.
import L from 'leaflet'

window.L = L
export default L
