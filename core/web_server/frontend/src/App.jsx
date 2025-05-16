import React, { useState, useEffect, useRef } from 'react';
import Box from '@mui/material/Box';
import Search from "./Search.jsx";
import Starmap from "./Starmap.jsx";
import MosaicPanel from "./MosaicPanel.jsx";
import TargetPanel from "./TargetPanel.jsx";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import RecentPanel from "./RecentPanel.jsx";
import SsePanel from "./SsePanel.jsx";
import FlatCorrectionPanel from "./FlatCorrectionPanel.jsx";
import logo from './logo.svg';
import { usePageVisibility } from './usePageVisibility';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { StyledEngineProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { createContext, useContext } from 'react';
import AstroFlowContext from './AstroFlowContext.jsx';
import AstroFlowBreadcrumbs from "./AstroFlowBreadcrumbs.jsx";
import { Stack } from '@mui/material';
import ProgressComponent from './ProgressComponent.jsx';
import WorkflowGraph from './WorkflowGraph.jsx';
import TileViewer from './TileViewer.jsx';
import StackViewer from './StackViewer.jsx';
import AppLayout from './AppLayout.jsx';
import AstroFlowImage from './AstroFlowImage.jsx';
import { FolderProvider } from './FolderContext';

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

  useEffect(() => {
    console.log('in App.useEffect');

    const loadObjectData = async () => {
      let [s, c, t, m, cart, cg] = await Promise.all([
        fetch('/stacks').then(r => r.json()),
        fetch('/calibration_stacks').then(r => r.json()),
        fetch('/targets').then(r => r.json()),
        fetch('/mosaics').then(r => r.json()),
        fetch(`/get_shopping_cart`).then(r => r.json()),
        fetch('/compute_graph').then(r => r.json()),
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
      let [s] = await Promise.all([
        fetch('/compute_graph_status').then(r => r.json()),
      ]);

      setComputeGraphStatus(s);
    };

    const loadRayStatus = async () => {
      console.log('loadRayStatus');

      let [s] = await Promise.all([
        fetch('/ray_status').then(r => r.json()),
      ]);

      setRayStatus(s);
    };

    loadComputeGraphStatus();
    loadRayStatus();

    // TODO: This removes the hostname's port, which is a little gross.
    const stream_url = window.location.protocol + '//' + window.location.hostname + '/stream';
    console.log('SSE stream URL:', stream_url);
    const sse = new window.EventSource(stream_url);

    sse.onmessage = e => loadObjectData();
    sse.addEventListener("reload_data", e => loadObjectData());
    sse.addEventListener("compute_graph_status", e => loadComputeGraphStatus());
    sse.addEventListener("ray_status", e => loadRayStatus());

    sse.onerror = () => {
      console.log('sse error');
      sse.close();
    }

    return () => {
      console.log('closing sse');
      sse.close();
    };
  }, []);

  // TODO: Breadcrumbs should preserve the path the user took to get here,
  // but with some normalization. One example: if the user clicks on a mosaic,
  // and then on a panel, the breadcrumb should reflect that path.
  //
  // This could maybe be implemented with "complex but normalized" routes like
  // /mosaic/:mosaic_id/target/:target_id/stack/:stack_id.

  // TODO: It's a little awkward to have to put <Search /> in each of these
  // routes.
  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <AstroFlowContext.Provider
          value={{ stacks, calibrationStacks, targets, mosaics, shoppingCart, computeGraph, computeGraphStatus, rayStatus }}
        >
          <FolderProvider>
            <CssBaseline />
            <Router>
              <Routes>
                <Route exact path="/" element={
                  <AppLayout>
                    <Starmap />
                  </AppLayout>
                } />
                <Route exact path="/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route exact path="/target/:target_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <TargetPanel />
                  </AppLayout>
                } />
                <Route exact path="/mosaic/:mosaic_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <MosaicPanel />
                  </AppLayout>
                } />
                <Route exact path="/target/:target_id/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route exact path="/mosaic/:mosaic_id/target/:target_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <TargetPanel />
                  </AppLayout>}
                />
                <Route exact path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <StackViewer />
                  </AppLayout>
                } />
                <Route exact path="/disambiguation/" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <DisambiguationPanel />
                  </AppLayout>
                } />
                <Route exact path="/disambiguation/:items" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <DisambiguationPanel />
                  </AppLayout>
                } />
                <Route exact path="/tile_viewer/:objectType/:imageHash" element={
                  <Box sx={{ width: '100vw', height: '100vh' }}>
                    <TileViewer />
                  </Box>
                } />
                <Route exact path="/recent" element={
                  dataIsAvailable &&
                  <AppLayout>
                    <RecentPanel />
                  </AppLayout>
                } />
                <Route exact path="/sse" element={
                  dataIsAvailable && <SsePanel />}
                />
                <Route exact path="/progress" element={
                  <ProgressComponent />}
                />
                <Route exact path="/workflow_graph" element={
                  dataIsAvailable && <WorkflowGraph theme={darkTheme} />}
                />
              </Routes>
            </Router>
          </FolderProvider>
        </AstroFlowContext.Provider>
      </StyledEngineProvider>
    </ThemeProvider >
  );
}

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#cccccc',
    },
  },
});

export default App;
