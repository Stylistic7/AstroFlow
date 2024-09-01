import React, { useState, useEffect, useRef } from 'react';
import Search from "./Search.jsx";
import Starmap from "./Starmap.jsx";
import TargetTable from "./TargetTable.jsx";
import StackTable from "./StackTable.jsx";
import MosaicTable from "./MosaicTable.jsx";
import MosaicPanel from "./MosaicPanel.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackPanel from "./StackPanel.jsx";
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
import { Stack } from '@mui/material';

function App() {
  const isPageVisible = usePageVisibility();
  const [dataIsAvailable, setDataIsAvailable] = useState(false);

  const [stacks, setStacks] = useState({});
  const [targets, setTargets] = useState({});
  const [mosaics, setMosaics] = useState({});
  const [shoppingCart, setShoppingCart] = useState([]);

  const removeKeys = (obj, keys) => {
    keys.forEach(key => {
      delete obj[key];
    });
  };

  useEffect(() => {
    console.log('in App.useEffect');
    const loadData = async () => {
      let [s, t, m, cart] = await Promise.all([
        fetch('/stacks').then(r => r.json()),
        fetch('/targets').then(r => r.json()),
        fetch('/mosaics').then(r => r.json()),
        fetch(`/get_shopping_cart`).then(r => r.json()),
      ]);

      setStacks(s);
      setTargets(t);
      setMosaics(m);
      setShoppingCart(cart);

      setDataIsAvailable(true);
    };

    loadData();

    // TODO: This removes the hostname's port, which is a little gross.
    const stream_url = window.location.protocol + '//' + window.location.hostname + '/stream';
    console.log('SSE stream URL:', stream_url);
    const sse = new window.EventSource(stream_url);
    function getRealtimeData(d) {
      console.log('Reloading data in response to SSE');
      loadData();
    }
    sse.onmessage = e => getRealtimeData(e.data);
    sse.addEventListener("reload_data", e => getRealtimeData(e.data));
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
  return (
    <ThemeProvider theme={darkTheme}>
      <StyledEngineProvider injectFirst>
        <AstroFlowContext.Provider
          value={{ stacks, targets, mosaics, shoppingCart }}
        >
          <CssBaseline />
          <Router>
            <Search />
            <Routes>
              <Route exact path="/" element={
                <Starmap />
              } />
              <Route exact path="/stack/:stack_id" element={
                dataIsAvailable && <StackPanel />}
              />
              <Route exact path="/target/:target_id" element={
                dataIsAvailable && <TargetPanel />}
              />
              <Route exact path="/mosaic/:mosaic_id" element={
                dataIsAvailable && <MosaicPanel />}
              />
              <Route exact path="/target/:target_id/stack/:stack_id" element={
                dataIsAvailable && <StackPanel />}
              />
              <Route exact path="/mosaic/:mosaic_id/target/:target_id" element={
                dataIsAvailable && <TargetPanel />}
              />
              <Route exact path="/mosaic/:mosaic_id/target/:target_id/stack/:stack_id" element={
                dataIsAvailable && <StackPanel />}
              />
              <Route exact path="/flat_correction/:stack_id" element={
                dataIsAvailable && <FlatCorrectionPanel stacks={stacks} />}
              />
              <Route exact path="/disambiguation/:items" element={
                dataIsAvailable && <DisambiguationPanel />}
              />
              <Route exact path="/recent" element={
                dataIsAvailable && <RecentPanel />}
              />
              <Route exact path="/sse" element={
                dataIsAvailable && <SsePanel />}
              />
            </Routes>
          </Router>
        </AstroFlowContext.Provider>
      </StyledEngineProvider>
    </ThemeProvider >
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
