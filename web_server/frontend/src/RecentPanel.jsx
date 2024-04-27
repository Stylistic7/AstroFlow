import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import MosaicCard from "./MosaicCard.jsx";
import TargetCard from "./TargetCard.jsx";
import DisambiguationPanel from "./DisambiguationPanel.jsx";

export default function RecentPanel({ stacks, targets, mosaics }) {
  const [recent, setRecent] = React.useState(undefined);

  const fetchRecent = () => {
    console.log('fetchRecent');

    fetch(`/recent`)
      .then((response) => {
        return response.json();
      })
      .then((j) => {
        let recentString = j.map(s => `target,${s}`).join(";");
        console.log('recentString', recentString);
        setRecent(recentString);
      });
  }

  useEffect(() => {
    fetchRecent();
  }, []);

  return (
    recent !== undefined && <DisambiguationPanel
      stacks={stacks}
      targets={targets}
      mosaics={mosaics}
      items_in={recent}
    />
  );
}


