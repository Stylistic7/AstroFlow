import React, { useState, useEffect } from 'react';
import MyTable from "./MyTable.js";
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
      <MyTable></MyTable>
    </ThemeProvider>
  );
}

export default App;
