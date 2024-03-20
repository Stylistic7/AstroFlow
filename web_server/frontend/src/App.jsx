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
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ReactPhotoSphereViewer } from 'react-photo-sphere-viewer';
import { MarkersPlugin } from '@photo-sphere-viewer/markers-plugin';
import "@photo-sphere-viewer/markers-plugin/index.css";

function App() {
  const isPageVisible = usePageVisibility();
  const timerIdRef = useRef(null);
  const [isPollingEnabled, setIsPollingEnabled] = useState(true);
  const [dataIsAvailable, setDataIsAvailable] = useState(false);

  const [stacks, setStacks] = useState({});
  const [targets, setTargets] = useState({});
  const [mosaics, setMosaics] = useState({});

  const [constellationPolylines, setConstellationPolylines] = useState([]);


  useEffect(() => {
    const pollingCallback = async () => {
      // Your polling logic here
      console.log('Polling...');

      // // Simulating an API failure in the polling callback
      // const shouldFail = Math.random() < 0.2; // Simulate 20% chance of API failure

      // if (shouldFail) {
      //   setIsPollingEnabled(false);
      //   console.log('Polling failed. Stopped polling.');
      // }

      // TODO: Only fetch the constellation polylines ones.
      let [c, s, t, m] = await Promise.all([
        fetch('/constellation_polylines').then(r => r.json()),
        fetch('/stacks').then(r => r.json()),
        fetch('/targets').then(r => r.json()),
        fetch('/mosaics').then(r => r.json()),
      ]);

      setConstellationPolylines(c);
      setStacks(s);
      setTargets(t);
      setMosaics(m);

      setDataIsAvailable(true);
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

  // const isEmptyObj = (obj) => {
  //   return Object.keys(obj).length === 0 && obj.constructor === Object;
  // };

  // const dataIsAvailable = !isEmptyObj(stacks) && !isEmptyObj(targets) && !isEmptyObj(mosaics);

  // const starmap = <Starmap
  //   constellation_polylines={constellationPolylines}
  //   stacks={stacks}
  //   targets={targets}
  //   mosaics={mosaics} />;

  const plugins = [
    [MarkersPlugin, {
      markers: [
        {
          id: "imageLayer2",
          image: "https://media.istockphoto.com/id/1464113334/photo/labrador-puppy.webp?s=2048x2048&w=is&k=20&c=CdRT4x1GS9T9xfxzs5YM09RUGUWkyfZY4x7T6Murn34=",
          size: { width: 220, height: 220 },
          position: { yaw: '130.5deg', pitch: '-0.1deg' },
          tooltip: "Image embedded in the scene"
        },
        {
          id: 'foobar',
          polygon: [
            [6.2208, 0.0906], [0.0443, 0.1028], [0.2322, 0.0849], [0.4531, 0.0387],
            [0.5022, -0.0056], [0.4587, -0.0396], [0.252, -0.0453], [0.0434, -0.0575],
            [6.1302, -0.0623], [6.0094, -0.0169], [6.0471, 0.032], [6.2208, 0.0906],
          ],
          style: {
            stroke: 'red',
            strokeWidth: '2px',
          },
          tooltip: {
            content: 'Mosaic NGC1977',
            position: 'bottom right',
          },
        },
      ],
    }],
  ]

  const photo_sphere_viewer = <ReactPhotoSphereViewer
    src="/static/images/ESA_Gaia_DR2_AllSky_Brightness_Colour_Cartesian_4000x2000.png"
    height={'100vh'}
    width={"100%"}
    plugins={plugins}>
  </ReactPhotoSphereViewer>;


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
            <Route exact path="/" element={photo_sphere_viewer} />
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
