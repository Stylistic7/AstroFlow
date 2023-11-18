import React, { useState, useEffect, useRef } from 'react';
import LeftDrawer from "./LeftDrawer.jsx";
import Search from "./Search.jsx";
import Starmap from "./Starmap.jsx";
import TargetTable from "./TargetTable.jsx";
import StackTable from "./StackTable.jsx";
import MosaicTable from "./MosaicTable.jsx";
import logo from './logo.svg';
import { usePageVisibility } from './usePageVisibility';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { StyledEngineProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import './App.css';

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

  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <CssBaseline />
        <Search stacks={stacks} targets={targets} mosaics={mosaics} />
        <LeftDrawer />
        <Starmap
          constellation_polylines={constellationPolylines}
          stacks={stacks}
          targets={targets}
          mosaics={mosaics} />
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
