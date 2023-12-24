import React, { useState, useEffect, useRef } from 'react';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import LeftDrawer from "./LeftDrawer.jsx";
import Search from "./Search.jsx";
import Starmap from "./Starmap.jsx";
import TargetTable from "./TargetTable.jsx";
import StackTable from "./StackTable.jsx";
import MosaicTable from "./MosaicTable.jsx";
import MosaicPanel from "./MosaicPanel.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackPanel from "./StackPanel.jsx";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import FlatCorrectionPanel from "./FlatCorrectionPanel.jsx";
import logo from './logo.svg';
import { usePageVisibility } from './usePageVisibility';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { StyledEngineProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import './App.css';
import { BrowserRouter as Router, Routes, Route} from 'react-router-dom';

function App() {
  const isPageVisible = usePageVisibility();
  const timerIdRef = useRef(null);
  const [isPollingEnabled, setIsPollingEnabled] = useState(true);

  const [stacks, setStacks] = useState({});
  const [targets, setTargets] = useState({});
  const [mosaics, setMosaics] = useState({});

  const [constellationPolylines, setConstellationPolylines] = useState([]);


  useEffect(() => {
    const pollingCallback = () => {
      // Your polling logic here
      console.log('Polling...');

      // // Simulating an API failure in the polling callback
      // const shouldFail = Math.random() < 0.2; // Simulate 20% chance of API failure

      // if (shouldFail) {
      //   setIsPollingEnabled(false);
      //   console.log('Polling failed. Stopped polling.');
      // }

      // TODO: Only fetch the constellation polylines ones.
      fetch('/constellation_polylines').then(res => res.json()).then(c => {
        setConstellationPolylines(c);  
      });
      fetch('/stacks').then(res => res.json()).then(stacks => {
        setStacks(stacks);  
      });
      fetch('/targets').then(res => res.json()).then(targets => {
        setTargets(targets);  
      });
      fetch('/mosaics').then(res => res.json()).then(mosaics => {
        setMosaics(mosaics);  
      });
    };

    const startPolling = () => {
      // pollingCallback(); // To immediately start fetching data
      // Polling every 30 seconds
      timerIdRef.current = setInterval(pollingCallback, 10000);
    };

    const stopPolling = () => {
      clearInterval(timerIdRef.current);
    };

    if (isPageVisible && isPollingEnabled) {
      pollingCallback();
      startPolling();
    } else {
      stopPolling();
    }

    return () => {
      stopPolling();
    };
  }, [isPageVisible, isPollingEnabled]);

  const isEmptyObj = (obj) => {
    return Object.keys(obj).length === 0 && obj.constructor === Object;
  };

  const dataIsAvailable = !isEmptyObj(stacks) && !isEmptyObj(targets) && !isEmptyObj(mosaics);

  const starmap = <Starmap
          constellation_polylines={constellationPolylines}
          stacks={stacks}
          targets={targets}
          mosaics={mosaics} />;

  const fontColor = "#aaa";

  // TODO: Breadcrumbs should preserve the path the user took to get here,
  // but with some normalization. One example: if the user clicks on a mosaic,
  // and then on a panel, the breadcrumb should reflect that path.
  //
  // This could maybe be implemented with "complex but normalized" routes like
  // /mosaic/:mosaic_id/target/:target_id/stack/:stack_id.
  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <CssBaseline />
        <Search stacks={stacks} targets={targets} mosaics={mosaics} />
        <LeftDrawer />
        <Router>
          <Routes>
            <Route exact path="/" element={starmap} />
            <Route exact path="/mosaic/:mosaic_id" element={
              dataIsAvailable && <MosaicPanel
                stacks={stacks} targets={targets} mosaics={mosaics}
              />}
            />
            <Route exact path="/target/:target_id" element={
              dataIsAvailable && <TargetPanel
                stacks={stacks} targets={targets} mosaics={mosaics}
              />}
            />
            <Route exact path="/stack/:stack_id" element={
              dataIsAvailable && <StackPanel
                stacks={stacks} targets={targets} mosaics={mosaics}
              />}
            />
            <Route exact path="/flat_correction/:stack_id" element={
              dataIsAvailable && <FlatCorrectionPanel stacks={stacks} />}
            />
            <Route exact path="/disambiguation/:items" element={
              dataIsAvailable && <DisambiguationPanel
                stacks={stacks} targets={targets} mosaics={mosaics}
              />}
            />
          </Routes>
        </Router>

      </StyledEngineProvider>
    </ThemeProvider>
  );
}

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
  components: {
    // Name of the component
    MuiTableCell: {
      styleOverrides: {
        // Name of the slot
        root: {
          // Some CSS
          backgroundColor: '#222',
        },
      },
    },
  },
});

export default App;
