import React, { useCallback, useRef, useEffect, useState, useContext } from 'react';

import { useNavigate, useLocation } from "react-router-dom";
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import RunningWithErrorsIcon from '@mui/icons-material/RunningWithErrors';
import { styled } from '@mui/system';

import { Box } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import FavoriteIcon from '@mui/icons-material/Favorite';
import PersonIcon from '@mui/icons-material/Person';


import Polyline from "./Polyline.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";

const strokeWidth = 0.0001;

const IconRowInSVG = React.forwardRef(function IconRowInSVG(props, ref) {
  const { icons, ...otherProps } = props;

  const iconRefs = useRef([]); // Array of refs
  const [positions, setPositions] = useState([]);

  const gap = 0; // Gap between icons

  useEffect(() => {
    // Ensure the array of refs is the same size as the array of icons
    iconRefs.current = iconRefs.current.slice(0, icons.length);
    iconRefs.current = icons.map((_, i) => iconRefs.current[i] || React.createRef());

    // Wait until the icons are rendered in the DOM
    const widths = iconRefs.current.map(ref => ref.current?.getBBox().width || 0);

    // Calculate positions based on widths
    let currentX = 0;
    const newPositions = widths.map(width => {
      const x = currentX;
      currentX += width + gap;
      return x;
    });

    setPositions(newPositions);
  }, [icons]);

  return (
    <svg width="0.04" height="0.005" xmlns="http://www.w3.org/2000/svg">
      <g ref={ref}>
        {icons.map((IconComponent, index) => (
          <g
            key={index}
            ref={iconRefs.current[index]}
            transform={`translate(${positions[index] || 0}, 0)`}
          >
            <IconComponent fontSize="large" />
          </g>
        ))}
      </g>
    </svg>
  );
});

function TargetOutline({ index, target_id, filter, projection, outline_visible, indices_visible, exposure_widgets_visible }) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);

  const navigate = useNavigate();
  const location = useLocation();

  const [svgElement, setSvgElement] = useState(null);

  const handleRef = useCallback((node) => {
    // console.log('handleRef node', node);
    if (node !== null) {
      setSvgElement(node);
    }
  }, []);

  useEffect(() => {
    if (svgElement) {
      const bbox = svgElement.getBBox();
      console.log('bbox.width', bbox.width);
      // svgElement.setAttribute('transform', `translate(${-bbox.width / 2}, ${0})`);
      svgElement.setAttribute('transform', `translate(${-0.02}, ${0})`);
    }
  }, [svgElement]);

  let stack_id = targets[target_id].stack_ids[0];
  if (filter) {
    stack_id = targets[target_id].stack_ids.find(
      s => stacks[s].filter == filter);
  }

  const middle_coord = coords => {
    const xs = coords.map(([x, y]) => x);
    const ys = coords.map(([x, y]) => y);

    return [
      (Math.min(...xs) + Math.max(...xs)) / 2.0,
      (Math.min(...ys) + Math.max(...ys)) / 2.0
    ];
  }

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

  const target_outline_style = {
    stroke: "#5555ee",
    strokeWidth: strokeWidth,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "visible"
  };

  const stack = stacks[stack_id];
  const coords = targets[target_id].projected_footprint[projection].coords;
  const [mean_x, mean_y] = middle_coord(coords)
  const width = 0.005;
  const height = 0.005;

  const no_stack = !stack.state['stack_files'];
  const too_few_frames = stack.state['num_frames'] < 3;

  const icons = [HomeIcon, FavoriteIcon, PersonIcon];

  const icon_spacing = 0.01;
  const num_icons = 1 + too_few_frames + no_stack;
  const icon_layout_width = num_icons * width + (num_icons - 1) * icon_spacing;

  return <>
    {outline_visible && <Polyline
      polyline={coords}
      key={target_id}
      style={target_outline_style}
      handleClick={event => handleClick(event, target_id)}
    />}
    <g transform={`translate(${mean_x - width / 2}, ${mean_y - height / 2})`}>
      {indices_visible && <text
        x="0"
        y="-0.002"
        fontFamily="Arial"
        fontSize="0.003"
        fill="white"
        letterSpacing="0"
        textLength="2"
        textAnchor="middle"
        style={{ pointerEvents: "none" }}
      >
        {index + 1}
      </text>}
      <IconRowInSVG ref={handleRef} icons={icons} />
    </g>
  </>
}


export default function MosaicOutline({
  mosaic_id,
  filter,
  outline_visible = true,
  indices_visible = false,
  exposure_widgets_visible = false,
  style = {}
}) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);
  const mosaic = mosaics[mosaic_id];

  const projection = 'mercator';

  const mosaic_outline_style = {
    stroke: "red",
    strokeWidth: strokeWidth,
    fillOpacity: 0,
    cursor: "pointer",
    pointerEvents: "none"
  };

  const RotatedSvg = styled('svg')({
    // transform: 'rotate(180deg)',
    // transformOrigin: 'center',
    // transform: 'scale(0.87, 1) translate(-60px, 0px)',
  });

  return (
    <RotatedSvg
      id="mosaic_svg"
      viewBox={mosaic.projected_footprint[projection].viewbox.join(" ")}
      style={{
        pointerEvents: "auto",
        margin: "2px",
        ...style
      }}
    >
      {
        mosaic.target_ids.map((target_id, index) =>
          <TargetOutline
            key={target_id}
            {...{ index, target_id, filter, projection, outline_visible, indices_visible, exposure_widgets_visible }}
          />
        )
      }
      {
        outline_visible && <Polyline
          polyline={mosaic.projected_footprint[projection].coords}
          key={mosaic_id}
          style={mosaic_outline_style}
        />
      }
    </RotatedSvg >
  );
}