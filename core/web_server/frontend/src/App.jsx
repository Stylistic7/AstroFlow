import { useState, useEffect, useMemo, useCallback } from 'react';
import Box from '@mui/material/Box';
import Starmap from "./Starmap.jsx";
import TargetPanel from "./TargetPanel.jsx";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import RecentPanel from "./RecentPanel.jsx";
import RegistryPanel from "./RegistryPanel.jsx";
import SsePanel from "./SsePanel.jsx";
import { usePageVisibility } from './usePageVisibility';
import { ThemeProvider } from '@mui/material/styles';
import { StyledEngineProvider } from '@mui/material/styles';
import darkTheme from './theme.js';
import CssBaseline from '@mui/material/CssBaseline';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AstroFlowContext from './AstroFlowContext.jsx';
import ProgressComponent from './ProgressComponent.jsx';
import WorkflowGraph from './WorkflowGraph.jsx';
import TileViewerV2 from './tile_viewer/TileViewerV2.jsx';
import StackViewer from './StackViewer.jsx';
import MosaicViewer from './MosaicViewer.jsx';
import MosaicDashboard from './MosaicDashboard.jsx';
import AppLayout from './AppLayout.jsx';
import PerformanceExplorer from './PerformanceExplorer.jsx';

import { FolderProvider } from './FolderContext';
import constellationPolylines from './constellationPolylines.json';
import PlannerPanel from './planner/PlannerPanel.jsx';
import DustCorrectionPanel from './DustCorrectionPanel.jsx';
import StoragePanel, { RequireFolders } from './StoragePanel.jsx';
import ToolsPanel from './ToolsPanel.jsx';
import SkyFlatsPanel from './SkyFlatsPanel.jsx';

function App() {
  const isPageVisible = usePageVisibility();
  const [dataIsAvailable, setDataIsAvailable] = useState(false);

  const [stacks, setStacks] = useState({});
  const [calibrationStacks, setCalibrationStacks] = useState({});
  const [targets, setTargets] = useState({});
  const [mosaics, setMosaics] = useState({});
  const [shoppingCart, setShoppingCart] = useState([]);
  const [plannerStars, setPlannerStars] = useState([]);    // [type, id] cart entries the planner shoots first
  const [archive, setArchive] = useState([]);
  const [catalogEntries, setCatalogEntries] = useState([]);
  const [computeGraph, setComputeGraph] = useState({});
  const [computeGraphStatus, setComputeGraphStatus] = useState({});
  const [rayStatus, setRayStatus] = useState({});
  const [asiairDevices, setAsiairDevices] = useState({ rigs: [], summary: { total: 0, active: 0 } });
  const [decisionOverrides, setDecisionOverrides] = useState([]);
  // Stack ids that are night-sky flat sources NOW: promoted by hand, or shot
  // as SkyFlat panes and not demoted (`GET /api/sky_flat_sources`).
  const [skyFlatSources, setSkyFlatSources] = useState([]);
  const [observingPlan, setObservingPlan] = useState({ entries: [] });
  const [sseConnected, setSseConnected] = useState(false);
  const [showQuarantined, setShowQuarantined] = useState(
    () => localStorage.getItem('astroflow_showQuarantined') !== 'false'
  );
  const [showForeign, setShowForeign] = useState(
    () => localStorage.getItem('astroflow_showForeign') !== 'false'
  );

  // Lifted out of the SSE useEffect so the AstroFlowContext provider
  // can expose `refreshAsiairDevices` to the panel — the panel calls
  // it after a successful PATCH to repaint immediately without
  // waiting for the SSE round-trip.
  const refreshAsiairDevices = useCallback(async () => {
    try {
      const s = await fetch('/api/asiair/devices').then(r => r.json());
      setAsiairDevices(s);
    } catch (e) { console.warn(e); }
  }, []);


  useEffect(() => {
    const loadObjectData = async () => {
      // Every fetch is individually guarded: one failing endpoint must
      // degrade ONE panel, never blank the whole UI. (A leftover fetch of
      // a removed endpoint once rejected this entire Promise.all and
      // emptied the starmap + cart despite a fully healthy backend.)
      const guarded = (url, fallback) =>
        fetch(url).then(r => r.ok ? r.json() : fallback).catch(() => fallback);
      const [s, c, t, m, cart, arch, catalog, stars] = await Promise.all([
        guarded('/api/stacks', {}),
        guarded('/api/calibration_stacks', {}),
        guarded('/api/targets', {}),
        guarded('/api/mosaics', {}),
        guarded('/api/shopping_cart', []),
        guarded('/api/archive', []),
        guarded('/api/catalog/entries', []),
        guarded('/api/planner/stars', []),
      ]);

      setStacks(s);
      setCalibrationStacks(c);
      setTargets(t);
      setMosaics(m);
      setShoppingCart(cart);
      setPlannerStars(stars);
      setArchive(arch);
      setCatalogEntries(catalog);

      setDataIsAvailable(true);
    };

    loadObjectData();

    // Polled on every SSE tick. A refused fetch (the browser's in-flight
    // budget during a zoom burst, a proxy hiccup) keeps the last value; it
    // must never surface as an unhandled rejection, which is a full-page
    // error overlay in the dev server (2026-10-01).
    const loadComputeGraphStatus = async () => {
      try {
        const s = await fetch('/api/compute_graph/status').then(r => r.json());
        setComputeGraphStatus(s);
      } catch (e) { console.warn(e); }
    };

    const loadRayStatus = async () => {
      try {
        const s = await fetch('/api/ray/status').then(r => r.json());
        setRayStatus(s);
      } catch (e) { console.warn(e); }
    };


    const loadDecisionOverrides = async () => {
      try {
        const d = await fetch('/api/decision_overrides').then(r => r.json());
        setDecisionOverrides(d.overrides || []);
        setSkyFlatSources(await fetch('/api/sky_flat_sources').then(r => r.json()));
      } catch (e) { console.warn(e); }
    };

    loadComputeGraphStatus();
    loadRayStatus();
    refreshAsiairDevices();
    loadDecisionOverrides();

    const stream_url = `${window.location.protocol}//${window.location.hostname}:8001/stream`;
    const HEARTBEAT_TIMEOUT_MS = 45000;
    let sse = null;
    let reconnectTimer = null;
    let heartbeatTimer = null;
    let destroyed = false;

    const resetHeartbeatTimer = () => {
      clearTimeout(heartbeatTimer);
      setSseConnected(true);
      heartbeatTimer = setTimeout(() => {
        // No message (including heartbeat) received for 15s — assume dead.
        setSseConnected(false);
        if (sse) { sse.close(); sse = null; }
        reconnectTimer = setTimeout(connect, 3000);
      }, HEARTBEAT_TIMEOUT_MS);
    };

    const connect = () => {
      if (destroyed) return;
      sse = new window.EventSource(stream_url);

      sse.onopen = () => resetHeartbeatTimer();

      // Any event (including heartbeat) resets the dead-man timer.
      const onAnyEvent = () => resetHeartbeatTimer();

      sse.onmessage = e => { onAnyEvent(); loadObjectData(); loadDecisionOverrides(); };
      sse.addEventListener("reload_data", e => { onAnyEvent(); loadObjectData(); loadDecisionOverrides(); });
      sse.addEventListener("compute_graph_status", e => { onAnyEvent(); loadComputeGraphStatus(); });
      sse.addEventListener("ray_status", e => { onAnyEvent(); loadRayStatus(); });
      sse.addEventListener("asiair_devices", e => { onAnyEvent(); refreshAsiairDevices(); });
      sse.addEventListener("heartbeat", onAnyEvent);

      sse.onerror = () => {
        setSseConnected(false);
        sse.close();
        sse = null;
        clearTimeout(heartbeatTimer);
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      destroyed = true;
      clearTimeout(reconnectTimer);
      clearTimeout(heartbeatTimer);
      if (sse) sse.close();
    };
  }, []);

  // TODO: Breadcrumbs should preserve the path the user took to get here,
  // but with some normalization. One example: if the user clicks on a mosaic,
  // and then on a panel, the breadcrumb should reflect that path.
  //
  // This could maybe be implemented with "complex but normalized" routes like
  // /mosaic/:mosaic_id/target/:target_id/stack/:stack_id.

  const toggleShowQuarantined = useCallback(() => {
    setShowQuarantined(prev => {
      const next = !prev;
      localStorage.setItem('astroflow_showQuarantined', String(next));
      return next;
    });
  }, []);

  // Build a set of quarantined (objectType, objectId) from decision overrides.
  const quarantinedSet = useMemo(() => {
    const set = new Set();
    for (const { key, choice } of decisionOverrides) {
      if (key.decision_type === "quarantine" && choice === "exclude") {
        set.add(`${key.object_type}:${key.subject_id}`);
      }
    }
    return set;
  }, [decisionOverrides]);

  const isQuarantined = useCallback(
    (objectType, objectId) => {
      if (quarantinedSet.has(`${objectType}:${objectId}`)) return true;
      // A target is effectively quarantined if all its stacks are.
      if (objectType === "target" && targets[objectId]?.stack_ids?.length > 0) {
        return targets[objectId].stack_ids.every(
          sid => quarantinedSet.has(`stack:${sid}`)
        );
      }
      return false;
    },
    [quarantinedSet, targets]
  );

  const quarantineCount = quarantinedSet.size;

  const toggleShowForeign = useCallback(() => {
    setShowForeign(prev => {
      const next = !prev;
      localStorage.setItem('astroflow_showForeign', String(next));
      return next;
    });
  }, []);

  // Other people's masters (shared master folders) and everything made from
  // them: a target with any foreign stack, a mosaic with any such target.
  const isForeign = useCallback(
    (objectType, objectId) => {
      const target = (id) => (targets[id]?.stack_ids || []).some(sid => stacks[sid]?.foreign);
      if (objectType === "stack") return Boolean(stacks[objectId]?.foreign);
      if (objectType === "target") return target(objectId);
      if (objectType === "mosaic") return (mosaics[objectId]?.target_ids || []).some(target);
      return false;
    },
    [stacks, targets, mosaics]
  );

  const foreignCount = useMemo(
    () => Object.values(stacks).filter(s => s.foreign).length, [stacks]);

  // Targets the user keeps out of every mosaic (decision "mosaic_membership").
  const standaloneTargets = useMemo(() => new Set(
    decisionOverrides
      .filter(({ key, choice }) => key.decision_type === "mosaic_membership" && choice === "standalone")
      .map(({ key }) => key.subject_id)
  ), [decisionOverrides]);

  // Shared master folders each stack/target/mosaic is published into
  // (decision "sharing": subject = the folder). Opt-in; see SharingMenu.jsx.
  const sharingMarks = useMemo(() => {
    const marks = new Map();
    decisionOverrides
      .filter(({ key, choice }) => key.decision_type === "sharing" && choice === "shared")
      .forEach(({ key }) => {
        const k = `${key.object_type}/${key.parent_stack_id}`;
        if (!marks.has(k)) marks.set(k, new Set());
        marks.get(k).add(key.subject_id);
      });
    return marks;
  }, [decisionOverrides]);
  const sharedFolders = useCallback(
    (objectType, objectId) => sharingMarks.get(`${objectType}/${objectId}`) || new Set(),
    [sharingMarks]
  );

  const skyFlatSourceSet = useMemo(() => new Set(skyFlatSources), [skyFlatSources]);

  const isSkyFlatSource = useCallback(
    (stackId) => skyFlatSourceSet.has(stackId), [skyFlatSourceSet]
  );

  const refreshDecisions = useCallback(async () => {
    try {
      const d = await fetch('/api/decision_overrides').then(r => r.json());
      setDecisionOverrides(d.overrides || []);
      setSkyFlatSources(await fetch('/api/sky_flat_sources').then(r => r.json()));
    } catch (e) { console.warn(e); }
  }, []);

  const refreshObservingPlan = useCallback(async () => {
    try {
      const p = await fetch('/api/observing_plan').then(r => r.json());
      setObservingPlan(p);
    } catch (e) { console.warn(e); }
  }, []);

  useEffect(() => { refreshObservingPlan(); }, [refreshObservingPlan]);

  // TODO: It's a little awkward to have to put <Search /> in each of these
  // routes.
  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <AstroFlowContext.Provider
          value={{
            constellationPolylines,
            stacks,
            calibrationStacks,
            targets,
            mosaics,
            catalogEntries,
            shoppingCart,
            plannerStars,
            archive,
            computeGraph,
            computeGraphStatus,
            rayStatus,
            asiairDevices,
            refreshAsiairDevices,
            isQuarantined,
            standaloneTargets,
            sharedFolders,
            isSkyFlatSource,
            showQuarantined,
            toggleShowQuarantined,
            quarantineCount,
            isForeign,
            showForeign,
            toggleShowForeign,
            foreignCount,
            refreshDecisions,
            observingPlan,
            refreshObservingPlan,
            sseConnected,
          }}
        >
          <FolderProvider>
            <CssBaseline />
            <Router>
              <Routes>
                <Route path="/" element={
                  <AppLayout>
                    <RequireFolders><Starmap /></RequireFolders>
                  </AppLayout>
                } />
                <Route path="/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/stack/:stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/stack/:stack_id/calibration_stack/:calibration_stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/stack/:stack_id/calibration_stack/:calibration_stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/calibration_stack/:calibration_stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/calibration_stack/:calibration_stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/target/:target_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <TargetPanel />
                  </AppLayout>
                } />
                {/* A target's view route renders the same tap viewer mosaics
                    use — the colour tail gives targets the same deliverables. */}
                <Route path="/target/:target_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicViewer />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicDashboard />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicViewer />
                  </AppLayout>
                } />
                <Route path="/target/:target_id/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/target/:target_id/stack/:stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/target/:target_id/stack/:stack_id/calibration_stack/:calibration_stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/target/:target_id/stack/:stack_id/calibration_stack/:calibration_stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/target/:target_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <TargetPanel />
                  </AppLayout>}
                />
                <Route path="/mosaic/:mosaic_id/target/:target_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicViewer />
                  </AppLayout>}
                />
                <Route path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id/calibration_stack/:calibration_stack_id/:pipeline/:finish" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id/calibration_stack/:calibration_stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route path="/disambiguation/" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <DisambiguationPanel />
                  </AppLayout>
                } />
                <Route path="/disambiguation/:items" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <DisambiguationPanel />
                  </AppLayout>
                } />
                <Route path="/tile_viewer/:objectType/:imageHash" element={
                  <Box sx={{ width: '100vw', height: '100vh' }}>
                    <TileViewerV2 />
                  </Box>
                } />
<Route path="/pipelines" element={
                  <AppLayout>
                    <RegistryPanel key="pipelines" kind="pipelines" />
                  </AppLayout>
                } />
                <Route path="/finishes" element={
                  <AppLayout>
                    <RegistryPanel key="finishes" kind="finishes" />
                  </AppLayout>
                } />
                <Route path="/planner" element={
                  <AppLayout>
                    <PlannerPanel />
                  </AppLayout>
                } />
                <Route path="/dust-correction" element={
                  <AppLayout>
                    <DustCorrectionPanel />
                  </AppLayout>
                } />
                <Route path="/folders" element={
                  <AppLayout>
                    <StoragePanel />
                  </AppLayout>
                } />
                <Route path="/tools" element={
                  <AppLayout>
                    <ToolsPanel />
                  </AppLayout>
                } />
                <Route path="/sky-flats" element={
                  <AppLayout>
                    <SkyFlatsPanel />
                  </AppLayout>
                } />
                <Route path="/recent" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <RecentPanel />
                  </AppLayout>
                } />
                <Route path="/sse" element={
                  dataIsAvailable && <SsePanel />}
                />
                <Route path="/progress" element={
                  <ProgressComponent />}
                />
                <Route path="/view_workflow_graph" element={
                  dataIsAvailable && <WorkflowGraph theme={darkTheme} />}
                />
                <Route path="/performance" element={
                  <AppLayout>
                    <PerformanceExplorer />
                  </AppLayout>}
                />
              </Routes>
            </Router>
          </FolderProvider>
        </AstroFlowContext.Provider>
      </StyledEngineProvider>
    </ThemeProvider>
  );
}

export default App;
