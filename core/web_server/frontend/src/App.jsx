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
  const [invalidated, setInvalidated] = useState([]);
  const [showQuarantined, setShowQuarantined] = useState(
    () => localStorage.getItem('astroflow_showQuarantined') !== 'false'
  );


  useEffect(() => {
    const loadObjectData = async () => {
      const [s, c, t, m, cart, cg] = await Promise.all([
        fetch('/api/stacks').then(r => r.json()),
        fetch('/api/calibration_stacks').then(r => r.json()),
        fetch('/api/targets').then(r => r.json()),
        fetch('/api/mosaics').then(r => r.json()),
        fetch('/api/shopping_cart').then(r => r.json()),
        fetch('/api/compute_graph').then(r => r.json()),
      ]);

      setStacks(s);
      setCalibrationStacks(c);
      setTargets(t);
      setMosaics(m);
      setShoppingCart(cart);
      setComputeGraph(cg);

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

    const loadInvalidated = async () => {
      try {
        const d = await fetch('/api/invalidated').then(r => r.json());
        setInvalidated(d);
      } catch { /* ignore */ }
    };

    loadComputeGraphStatus();
    loadRayStatus();
    loadInvalidated();

    const stream_url = window.location.protocol + '//' + window.location.hostname + '/stream';
    const sse = new window.EventSource(stream_url);

    sse.onmessage = e => { loadObjectData(); loadInvalidated(); };
    sse.addEventListener("reload_data", e => { loadObjectData(); loadInvalidated(); });
    sse.addEventListener("compute_graph_status", e => loadComputeGraphStatus());
    sse.addEventListener("ray_status", e => loadRayStatus());

    sse.onerror = () => {
      sse.close();
    }

    return () => {
      sse.close();
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

  const filterQuarantined = useCallback((items) => {
    if (showQuarantined) return items;
    const result = {};
    for (const [k, v] of Object.entries(items || {})) {
      if (!v?.is_quarantined) result[k] = v;
    }
    return result;
  }, [showQuarantined]);

  const visibleStacks = useMemo(() => filterQuarantined(stacks), [stacks, filterQuarantined]);
  const visibleTargets = useMemo(() => filterQuarantined(targets), [targets, filterQuarantined]);
  const visibleMosaics = useMemo(() => filterQuarantined(mosaics), [mosaics, filterQuarantined]);

  const quarantineCount = useMemo(() => {
    const count = (items) => Object.values(items || {}).filter(v => v?.is_quarantined).length;
    return count(stacks) + count(targets) + count(mosaics);
  }, [stacks, targets, mosaics]);

  // TODO: It's a little awkward to have to put <Search /> in each of these
  // routes.
  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <AstroFlowContext.Provider
          value={{
            constellationPolylines,
            stacks: visibleStacks,
            calibrationStacks,
            targets: visibleTargets,
            mosaics: visibleMosaics,
            shoppingCart,
            computeGraph,
            computeGraphStatus,
            rayStatus,
            invalidated,
            showQuarantined,
            toggleShowQuarantined,
            quarantineCount,
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
