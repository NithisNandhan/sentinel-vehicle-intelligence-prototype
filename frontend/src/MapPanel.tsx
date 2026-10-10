import { useEffect, useRef, useState } from 'react';
import maplibregl, { type Map, type Marker } from 'maplibre-gl';
import type { Feature, Polygon } from 'geojson';
import type { Camera, Sighting } from './api';
import { sourceLabel } from './recognition';

interface Props {
  cameras: Camera[];
  route?: Sighting[];
  selectedCameraId?: string | null;
  onSelectCamera?: (cameraId: string) => void;
}

// A real street map (OpenStreetMap tiles) when the browser can reach it. The
// tiles are fetched by the browser from tile.openstreetmap.org, so the map area
// being viewed is visible to that service; no camera or plate data is sent.
// If the tiles cannot be loaded the panel falls back to the offline outline.
const streetStyle: maplibregl.StyleSpecification = {
  version: 8,
  name: 'Street map',
  sources: {
    streets: {
      type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 19,
      attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#0c1d2c' } },
    { id: 'streets', type: 'raster', source: 'streets' },
  ],
};

// Approximate Gujarat silhouette for geographic orientation when offline.
// This is a decorative guide, not an authoritative administrative boundary.
const gujaratOutline: Feature<Polygon> = {
  type: 'Feature', properties: {}, geometry: {
    type: 'Polygon', coordinates: [[
      [68.0, 23.1], [68.6, 23.8], [69.5, 24.3], [70.1, 24.6],
      [71.2, 24.3], [72.1, 24.6], [73.0, 24.4], [73.4, 23.4],
      [74.0, 22.5], [74.2, 21.8], [73.2, 20.9], [72.5, 20.2],
      [72.0, 21.0], [71.3, 21.0], [70.3, 20.8], [69.5, 21.5],
      [69.1, 22.4], [68.5, 22.4], [68.0, 23.1],
    ]],
  },
};

function located(camera: Camera): camera is Camera & { latitude: number; longitude: number } {
  return typeof camera.latitude === 'number' && Number.isFinite(camera.latitude)
    && typeof camera.longitude === 'number' && Number.isFinite(camera.longitude);
}

/** Info window content, built as DOM nodes so camera text is never parsed as HTML. */
function infoWindow(camera: Camera & { latitude: number; longitude: number }, open?: () => void): HTMLElement {
  const box = document.createElement('div');
  box.className = 'map-info';
  const title = document.createElement('strong');
  title.textContent = camera.name || camera.id;
  const rows: [string, string][] = [
    ['Camera', camera.id.toUpperCase()],
    ['Status', camera.status],
    ['Source', sourceLabel(camera.source_mode)],
    ['Area', camera.region || 'Not provided'],
    ['Position', `${camera.latitude.toFixed(4)}, ${camera.longitude.toFixed(4)}${camera.coordinates_approximate === false ? '' : ' (approximate)'}`],
  ];
  box.append(title);
  for (const [label, value] of rows) {
    const line = document.createElement('span');
    const name = document.createElement('em');
    name.textContent = label;
    line.append(name, document.createTextNode(value));
    box.append(line);
  }
  if (open) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Open camera details';
    button.addEventListener('click', open);
    box.append(button);
  }
  return box;
}

export default function MapPanel({ cameras, route = [], selectedCameraId, onSelectCamera }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const markersRef = useRef<Marker[]>([]);
  const [streets, setStreets] = useState(true);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: streetStyle,
      center: [72.57, 23.03],
      zoom: 11,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    map.addControl(new maplibregl.FullscreenControl(), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');
    map.on('load', () => {
      map.addSource('region', { type: 'geojson', data: gujaratOutline });
      map.addLayer({ id: 'region-fill', type: 'fill', source: 'region', layout: { visibility: 'none' }, paint: {
        'fill-color': '#17374a', 'fill-opacity': 0.38,
      } });
      map.addLayer({ id: 'region-outline', type: 'line', source: 'region', layout: { visibility: 'none' }, paint: {
        'line-color': '#426a80', 'line-width': 1.3, 'line-dasharray': [2, 2],
      } });
    });
    // No network, or the tile service is unreachable: show the offline outline instead of a blank panel.
    map.on('error', event => {
      if ((event as { sourceId?: string }).sourceId !== 'streets') return;
      setStreets(false);
      for (const layer of ['region-fill', 'region-outline']) {
        if (map.getLayer(layer)) map.setLayoutProperty(layer, 'visibility', 'visible');
      }
    });
    // The panel is laid out after the map is created; keep the canvas the size of its box.
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(containerRef.current);
    mapRef.current = map;
    return () => { observer.disconnect(); map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];
    const routePositions = route.map(sighting => {
      const camera = cameras.find(item => item.id === sighting.camera_id);
      const lat = sighting.latitude ?? camera?.latitude;
      const lon = sighting.longitude ?? camera?.longitude;
      return typeof lat === 'number' && typeof lon === 'number' ? [lon, lat] : null;
    }).filter((point): point is number[] => point !== null);
    const points = routePositions.length ? routePositions : cameras.filter(located).map(camera => [camera.longitude, camera.latitude]);

    cameras.filter(located).forEach(camera => {
      // A pin with the camera's name beside it, and an info window on click.
      const element = document.createElement('button');
      element.type = 'button';
      element.className = `map-pin ${selectedCameraId === camera.id ? 'selected' : ''}`;
      const dot = document.createElement('span');
      dot.className = `map-marker ${camera.status.toLowerCase()} ${camera.source_mode === 'synthetic' ? 'synthetic' : ''}`;
      const label = document.createElement('span');
      label.className = 'map-pin-label';
      label.textContent = camera.name || camera.id;
      element.append(dot, label);
      const source = camera.source_mode === 'live' ? '' : `${sourceLabel(camera.source_mode)} · `;
      element.setAttribute('aria-label', `${camera.name}, ${source}${camera.status}`);
      element.title = `${camera.name} · ${source}${camera.status}`;
      // Clicking a pin opens its info window, as on a consumer map; the details page is one more click.
      const popup = new maplibregl.Popup({ offset: 16, closeButton: true, maxWidth: '260px' })
        .setDOMContent(infoWindow(camera, onSelectCamera ? () => onSelectCamera(camera.id) : undefined));
      markersRef.current.push(new maplibregl.Marker({ element, anchor: 'left', offset: [-9, 0] })
        .setLngLat([camera.longitude, camera.latitude]).setPopup(popup).addTo(map));
    });

    if (map.getLayer('route-line')) map.removeLayer('route-line');
    if (map.getSource('route')) map.removeSource('route');
    if (routePositions.length >= 2) {
      const draw = () => {
        if (map.getSource('route')) return;
        map.addSource('route', { type: 'geojson', data: {
          type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: routePositions },
        } });
        map.addLayer({ id: 'route-line', type: 'line', source: 'route', layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: {
          'line-color': '#1a73e8', 'line-width': 5, 'line-opacity': 0.9,
        } });
      };
      if (map.isStyleLoaded()) draw(); else map.once('load', draw);
    }
    if (points.length) {
      map.resize();
      const bounds = new maplibregl.LngLatBounds();
      points.forEach(point => bounds.extend([point[0], point[1]]));
      // Street level, but never so close that two nearby cameras fill the whole view.
      map.fitBounds(bounds, { padding: 90, maxZoom: 14, duration: 500 });
    }
  }, [cameras, route, selectedCameraId, onSelectCamera]);

  const missing = cameras.filter(camera => !located(camera)).length;
  return <div className={`map-wrap ${streets ? 'streets' : ''}`}>
    <div ref={containerRef} className="map-canvas" aria-label="Street map of camera locations" />
    <div className="map-scale-note">{streets ? 'Camera positions are approximate demo coordinates' : 'Street map unavailable · offline outline with approximate positions'}</div>
    {missing > 0 && <div className="map-missing-note">{missing} source{missing === 1 ? ' has' : 's have'} no mapped coordinates</div>}
  </div>;
}
