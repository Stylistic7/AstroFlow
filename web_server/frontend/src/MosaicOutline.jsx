import React, { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Polyline from "./Polyline.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

export default function MosaicOutline({ mosaic_id, style = {} }) {
  const { targets, mosaics } = useContext(AstroFlowContext);
  const mosaic = mosaics[mosaic_id];
  console.log(mosaic);

  const navigate = useNavigate();

  const projection = 'mercator';

  const strokeWidth = 0.0002;

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
      viewBox={mosaic.projected_footprint[projection].viewbox.join(" ")}
      style={{
        pointerEvents: "auto",
        margin: "2px",
        ...style
      }}
    >
      {mosaic.target_ids.map((target_id) =>
        <Polyline
          polyline={targets[target_id].projected_footprint[projection].coords}
          key={target_id}
          style={target_outline_style}
          handleClick={event => navigate(`/target/${target_id}`)}
        />
      )}
      <Polyline
        polyline={mosaic.projected_footprint[projection].coords}
        key={mosaic_id}
        style={mosaic_outline_style}
      />
    </svg>
  );
}