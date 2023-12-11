import * as React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import Polyline from "./Polyline.jsx";

export default function MosaicOutline({stacks, targets, mosaics, mosaic_id}) {
  const mosaic = mosaics[mosaic_id];

  const target_outline_style = {
    stroke: "#5555ee",
    strokeWidth: 0.15,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "visible"
  };

  const mosaic_outline_style = {
    stroke: "red",
    strokeWidth: 0.15,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "visible"
  };

  return (
    <svg
      id="mosaic_svg"
      viewBox={mosaic.viewbox.join(" ")}
      style={{
        pointerEvents: "auto",
        margin: "2px",
        maxWidth: "100vw",
        maxHeight: "80vh",
    }}
    >
      {mosaic.target_ids.map((target_id) =>
        <Polyline
          polyline={targets[target_id].footprint}
          key={target_id}
          style={target_outline_style}
        />
      )}
      <Polyline
        polyline={mosaic.footprint}
        key={mosaic_id}
        style={mosaic_outline_style}
      />
    </svg>
  );
}