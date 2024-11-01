import React, { useContext } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import MoreTimeIcon from '@mui/icons-material/MoreTime';
import FilterNoneIcon from '@mui/icons-material/FilterNone';
import { styled } from '@mui/system';

import Polyline from "./Polyline.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";
import CircleSlashIcon from "./CircleSlashIcon.jsx";

function NoStackIcon({ x, y, width, height }) {
  return <CircleSlashIcon
    IconComponent={FilterNoneIcon}
    x={x}
    y={y}
    width={width}
    height={height}
  />
}

export default function MosaicOutline({
  mosaic_id,
  filter,
  width,
  height,
  outline_visible = true,
  indices_visible = false,
  exposure_widgets_visible = false,
  style = {}
}) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);
  const mosaic = mosaics[mosaic_id];

  const navigate = useNavigate();
  const location = useLocation();

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

  const RotatedSvg = styled('svg')({
    // transform: 'rotate(180deg)',
    // transformOrigin: 'center',
    // transform: 'scale(0.87, 1) translate(-60px, 0px)',
  });

  const handleClick = (event, target_id) => {
    const link = `${location.pathname}/target/${target_id}`;
    if (event.ctrlKey || event.metaKey) {
      // Open in a new tab if Ctrl (Windows/Linux) or Command (Mac) key is held
      window.open(link, '_blank');
    } else {
      // Otherwise, navigate normally
      navigate(link);
    }
  };

  return (
    <RotatedSvg
      id="mosaic_svg"
      width={width}
      height={height}
      viewBox={mosaic.projected_footprints[projection].viewbox.join(" ")}
      style={{
        pointerEvents: "auto",
        margin: "2px",
        ...style
      }}
    >
      {mosaic.target_ids.map((target_id, index) => {
        let stack_id = targets[target_id].stack_ids[0];
        if (filter) {
          stack_id = targets[target_id].stack_ids.find(
            s => stacks[s].filter == filter);
        }

        const stack = stacks[stack_id];
        const coords = targets[target_id].projected_footprints[projection].coords;
        const [mean_x, mean_y] = middle_coord(coords)
        const width = 0.005;
        const height = 0.005;

        // const too_few_frames = stack.state['num_frames'] < 3;
        // const no_stack = !stack.state['stack_files'];

        let error_icon = undefined;
        if (stack.state['num_frames'] < 3) {
          error_icon = <MoreTimeIcon
            x="0.0025"
            y="0"
            width={width}
            height={height}
          />
        } else if (!stack.state['stack_files']) {
          error_icon = <NoStackIcon
            x="0.0025"
            y="0"
            width={width}
            height={height}
          />
        }

        return <>
          {outline_visible && <Polyline
            polyline={coords}
            key={target_id}
            style={target_outline_style}
            handleClick={event => handleClick(event, target_id)}
          />}
          <g transform={`translate(${mean_x - width / 2}, ${mean_y - height / 2})`}>
            {indices_visible && <text
              x="0.002"
              y="-0.002"
              fontFamily="Arial"
              fontSize="0.003"
              fill="white"
              letter-spacing="0"
              text-length="2"
              text-anchor="middle"
              style={{ pointerEvents: "none" }}
            >
              {index + 1}
            </text>}
            {exposure_widgets_visible && filter && stack && <ExposureDurationWidget
              x={error_icon ? "-0.0025" : "0"}
              y="0.0008"
              width={width * 0.75}
              height={height * 0.75}
              key={`exposure_duration_${stack_id}`}
              totalExposureDuration={stack.integration_s / 3600.0}
              selectedExposureDuration={stack.integration_s / 3600.0}
              stack={stack}
              style={{ pointerEvents: "none" }}
            />}
            {error_icon}
          </g>
        </>
      })}
      {outline_visible && <Polyline
        polyline={mosaic.projected_footprints[projection].coords}
        key={mosaic_id}
        style={mosaic_outline_style}
      />}
    </RotatedSvg >
  );
}