// Generated from backend/app/unified_contract.py; do not edit.
export type CharacterAlternative = {
  character: string;
  score: number;
};

export type CharacterEvidence = {
  character: string;
  score: number;
  alternatives?: Array<CharacterAlternative>;
};

export type DiagnosticsSnapshot = {
  schema_version?: "1.0";
  source_mode?: "recorded";
  session_id: string;
  camera_id: string;
  video_position_ms?: number | null;
  models?: Record<string, string | null>;
  stage_ms?: Record<string, number>;
  track: TrackDiagnostics;
};

export type FrameEvidence = {
  pts_ms: number;
  width: number;
  height: number;
  quality?: number | null;
  text?: string;
  characters?: Array<CharacterEvidence>;
  image?: string | null;
};

export type LocalSessionOut = {
  csrf_token: string;
  expires_at: string;
};

export type PlateCandidate = {
  plate: string;
  score: number;
  minimum_character_score?: number | null;
  log_score?: number | null;
  observations?: number | null;
};

export type ReconstructionEvidence = {
  status: string;
  reason?: string;
  image?: string | null;
  text?: string | null;
  width?: number | null;
  height?: number | null;
  contributing_originals?: number;
  elapsed_ms?: number | null;
  characters?: Array<CharacterEvidence>;
  candidates?: Array<PlateCandidate>;
};

export type TrackDiagnostics = {
  track_id: string;
  vehicle_box?: Array<number>;
  vehicle_class?: string | number;
  seen_pts_ms?: number | null;
  observations?: number;
  recognition_state?: string;
  masked_plate?: string | null;
  best_reading?: string | null;
  model_score?: number | null;
  gate_reasons?: Array<string>;
  frames?: Array<FrameEvidence>;
  candidates?: Array<PlateCandidate>;
  reconstruction?: ReconstructionEvidence | null;
};

export type WatchlistCreate = {
  plate: string;
  reason: string;
  severity?: "low" | "medium" | "high" | "critical";
};

export type WatchlistItem = {
  id: number;
  masked_plate?: string | null;
  reason: string;
  severity: "low" | "medium" | "high" | "critical";
  active: boolean;
  version: number;
  created_at: string;
};

export type WatchlistPage = {
  items: Array<WatchlistItem>;
  total: number;
  offset: number;
  limit: number;
  version: number;
};

export type WatchlistPatch = {
  expected_version: number;
  reason?: string | null;
  severity?: "low" | "medium" | "high" | "critical" | null;
  active?: boolean | null;
};

export type WatchlistSearch = {
  query?: string;
  offset?: number;
  limit?: number;
};
