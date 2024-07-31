import React, { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Polyline from "./Polyline.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";

export default function MosaicOutline({ mosaic_id, filter, style = {} }) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);
  const mosaic = mosaics[mosaic_id];
  console.log(mosaic);
  console.log('filter', filter);

  const navigate = useNavigate();

  const projection = 'mercator';

  const strokeWidth = 0.0001;

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

  const middle_coord = coords => {
    const xs = coords.map(([x, y]) => x);
    const ys = coords.map(([x, y]) => y);

    return [
      (Math.min(...xs) + Math.max(...xs)) / 2.0,
      (Math.min(...ys) + Math.max(...ys)) / 2.0
    ];
  }

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
      {mosaic.target_ids.map((target_id) => {
        let stack_id = targets[target_id].stack_ids[0];
        if (filter) {
          stack_id = targets[target_id].stack_ids.find((stack_id) => stacks[stack_id].filter == filter);
        }

        const stack = stacks[stack_id];
        const coords = targets[target_id].projected_footprint[projection].coords;
        const [mean_x, mean_y] = middle_coord(coords)
        const width = 0.005;
        const height = 0.005;

        return <>
          <Polyline
            polyline={coords}
            key={target_id}
            style={target_outline_style}
            handleClick={event => navigate(`/target/${target_id}`)}
          />
          {filter && stack && <ExposureDurationWidget
            x={mean_x - width / 2}
            y={mean_y - height / 2}
            width={width}
            height={height}
            key={`exposure_duration_${stack_id}`}
            totalExposureDuration={stack.integration_s / 3600.0}
            selectedExposureDuration={stack.integration_s / 3600.0}
            stack={stack}
          />}
        </>
      })}
      <Polyline
        polyline={mosaic.projected_footprint[projection].coords}
        key={mosaic_id}
        style={mosaic_outline_style}
      />
    </svg >
  );
}