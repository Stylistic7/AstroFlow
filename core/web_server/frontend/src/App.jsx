import { useState, useEffect, useMemo, useCallback } from 'react';
import Box from '@mui/material/Box';
import Starmap from "./Starmap.jsx";
import TargetPanel from "./TargetPanel.jsx";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import RecentPanel from "./RecentPanel.jsx";
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
import TileViewer from './TileViewer.jsx';
import StackViewer from './StackViewer.jsx';
import MosaicViewer from './MosaicViewer.jsx';
import MosaicDashboard from './MosaicDashboard.jsx';
import AppLayout from './AppLayout.jsx';
import PerformanceExplorer from './PerformanceExplorer.jsx';

import { FolderProvider } from './FolderContext';
import constellationPolylines from './constellationPolylines.json';

function App() {
  const isPageVisible = usePageVisibility();
  const [dataIsAvailable, setDataIsAvailable] = useState(false);

  const [stacks, setStacks] = useState({});
  const [calibrationStacks, setCalibrationStacks] = useState({});
  const [targets, setTargets] = useState({});
  const [mosaics, setMosaics] = useState({});
  const [shoppingCart, setShoppingCart] = useState([]);
  const [computeGraph, setComputeGraph] = useState({});
  const [computeGraphStatus, setComputeGraphStatus] = useState({});
  const [rayStatus, setRayStatus] = useState({});
  const [decisionOverrides, setDecisionOverrides] = useState([]);
  const [validity, setValidity] = useState({});
  const [observingPlan, setObservingPlan] = useState({ entries: [] });
  const [sseConnected, setSseConnected] = useState(false);
  const [showQuarantined, setShowQuarantined] = useState(
    () => localStorage.getItem('astroflow_showQuarantined') !== 'false'
  );


  useEffect(() => {
    const loadObjectData = async () => {
      const [s, c, t, m, cart, cg, v] = await Promise.all([
        fetch('/api/stacks').then(r => r.json()),
        fetch('/api/calibration_stacks').then(r => r.json()),
        fetch('/api/targets').then(r => r.json()),
        fetch('/api/mosaics').then(r => r.json()),
        fetch('/api/shopping_cart').then(r => r.json()),
        fetch('/api/compute_graph').then(r => r.json()),
        fetch('/api/validity').then(r => r.json()).catch(() => ({})),
      ]);

      setStacks(s);
      setCalibrationStacks(c);
      setTargets(t);
      setMosaics(m);
      setShoppingCart(cart);
      setComputeGraph(cg);
      setValidity(v);

      setDataIsAvailable(true);
    };

    loadObjectData();

    const loadComputeGraphStatus = async () => {
      const s = await fetch('/api/compute_graph/status').then(r => r.json());
      setComputeGraphStatus(s);
    };

    const loadRayStatus = async () => {
      const s = await fetch('/api/ray/status').then(r => r.json());
      setRayStatus(s);
    };

    const loadDecisionOverrides = async () => {
      try {
        const d = await fetch('/api/decision_overrides').then(r => r.json());
        setDecisionOverrides(d.overrides || []);
      } catch (e) { console.warn(e); }
    };

    loadComputeGraphStatus();
    loadRayStatus();
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

  const refreshDecisions = useCallback(async () => {
    try {
      const d = await fetch('/api/decision_overrides').then(r => r.json());
      setDecisionOverrides(d.overrides || []);
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
            shoppingCart,
            computeGraph,
            computeGraphStatus,
            rayStatus,
            validity,
            isQuarantined,
            showQuarantined,
            toggleShowQuarantined,
            quarantineCount,
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
                    <Starmap />
                  </AppLayout>
                } />
                <Route path="/stack/:stack_id" element={
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
                <Route path="/mosaic/:mosaic_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicDashboard />
                  </AppLayout>
                } />
                <Route path="/mosaic/:mosaic_id/:implementation/:quality/:filter" element={
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
                <Route path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id" element={
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
                    <TileViewer />
                  </Box>
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
