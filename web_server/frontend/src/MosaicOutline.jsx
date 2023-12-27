import * as React from 'react';
import { useNavigate, useParams } from "react-router-dom";
import PropTypes from 'prop-types';

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import Polyline from "./Polyline.jsx";

export default function MosaicOutline({stacks, targets, mosaics, mosaic_id, style={}}) {
  const mosaic = mosaics[mosaic_id];

  const navigate = useNavigate();

  const [min_ra, min_dec, ra_span, dec_span] = mosaic.viewbox;
  const area = Math.abs(ra_span) * Math.abs(dec_span);
  console.log('area', area);

  const strokeWidth = 0.08 * (area / 180);

  const target_outline_style = {
    stroke: "#5555ee",
    strokeWidth: strokeWidth,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "visible"
  };

  const mosaic_outline_style = {
    stroke: "red",
    strokeWidth: strokeWidth,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "none"
  };

  return (
    <svg
      id="mosaic_svg"
      viewBox={mosaic.viewbox.join(" ")}
      style={{
        pointerEvents: "auto",
        margin: "2px",
        maxWidth: "100vw",
        maxHeight: "25vh",

        ...style
      }}
    >
      {mosaic.target_ids.map((target_id) =>
        <Polyline
          polyline={targets[target_id].footprint}
          key={target_id}
          style={target_outline_style}
          handleClick={event => navigate(`/target/${target_id}`)}
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