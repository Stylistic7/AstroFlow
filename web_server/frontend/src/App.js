import React, { useState, useEffect } from 'react';
import TargetTable from "./TargetTable.jsx";
import logo from './logo.svg';
import './App.css';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});


function App() {
  const [currentTime, setCurrentTime] = useState(0);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <TargetTable></TargetTable>
    </ThemeProvider>
  );
}

export default App;
