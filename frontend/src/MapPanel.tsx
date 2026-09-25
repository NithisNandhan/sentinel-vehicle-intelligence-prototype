import { useEffect, useRef } from 'react';
import maplibregl, { type Map, type Marker } from 'maplibre-gl';
import type { Feature, Polygon } from 'geojson';
import type { Camera, Sighting } from './api';

interface Props {
  cameras: Camera[];
  route?: Sighting[];
  selectedCameraId?: string | null;
  onSelectCamera?: (cameraId: string) => void;
}

const emptyStyle: maplibregl.StyleSpecification = {
  version: 8,
  name: 'Offline command map',
  sources: {},
  layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#0c1d2c' } }],
};

// Approximate Gujarat silhouette for geographic orientation in the offline demo.
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

export default function MapPanel({ cameras, route = [], selectedCameraId, onSelectCamera }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const markersRef = useRef<Marker[]>([]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: emptyStyle,
      center: [71.55, 22.65],
      zoom: 6.3,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    map.on('load', () => {
      map.addSource('region', { type: 'geojson', data: gujaratOutline });
      map.addLayer({ id: 'region-fill', type: 'fill', source: 'region', paint: {
        'fill-color': '#17374a', 'fill-opacity': 0.38,
      } });
      map.addLayer({ id: 'region-outline', type: 'line', source: 'region', paint: {
        'line-color': '#426a80', 'line-width': 1.3, 'line-dasharray': [2, 2],
      } });
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
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
      const element = document.createElement('button');
      element.type = 'button';
      element.className = `map-marker ${camera.status.toLowerCase()} ${camera.source_mode === 'synthetic' ? 'synthetic' : ''} ${selectedCameraId === camera.id ? 'selected' : ''}`;
      const sourceLabel = camera.source_mode === 'synthetic' ? 'SIMULATED · ' : '';
      element.setAttribute('aria-label', `${camera.name}, ${sourceLabel}${camera.status}`);
      element.title = `${camera.name} · ${sourceLabel}${camera.status}`;
      element.addEventListener('click', () => onSelectCamera?.(camera.id));
      markersRef.current.push(new maplibregl.Marker({ element })
        .setLngLat([camera.longitude, camera.latitude]).addTo(map));
    });

    if (map.getLayer('route-line')) map.removeLayer('route-line');
    if (map.getSource('route')) map.removeSource('route');
    if (routePositions.length >= 2) {
      const draw = () => {
        if (map.getSource('route')) return;
        map.addSource('route', { type: 'geojson', data: {
          type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: routePositions },
        } });
        map.addLayer({ id: 'route-line', type: 'line', source: 'route', paint: {
          'line-color': '#f6b75d', 'line-width': 3, 'line-opacity': 0.9,
        } });
      };
      if (map.isStyleLoaded()) draw(); else map.once('load', draw);
    }
    if (points.length) {
      const bounds = new maplibregl.LngLatBounds();
      points.forEach(point => bounds.extend([point[0], point[1]]));
      // Keep enough regional context when two nearby demo cameras are selected.
      map.fitBounds(bounds, { padding: 80, maxZoom: 7.2, duration: 500 });
    }
  }, [cameras, route, selectedCameraId, onSelectCamera]);

  const missing = cameras.filter(camera => !located(camera)).length;
  return <div className="map-wrap">
    <div ref={containerRef} className="map-canvas" aria-label="Approximate Gujarat camera map" />
    <div className="map-scale-note">Approximate demo map · offline basemap</div>
    {missing > 0 && <div className="map-missing-note">{missing} cameras have no mapped coordinates</div>}
  </div>;
}
