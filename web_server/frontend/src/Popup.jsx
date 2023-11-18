import React, { useState, useRef, useEffect, useReducer } from "react";
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import LayersIcon from '@mui/icons-material/Layers';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import MosaicPanel from "./MosaicPanel.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackPanel from "./StackPanel.jsx";
import "./Starmap.css";

export default function Popup({ mosaic_id, stack_id, mosaics, targets, stacks, zoomToElement }) {
	const [stack, setStack] = React.useState(undefined);

	if (!mosaic_id) {
		return;
	}

	const mosaic = mosaics[mosaic_id];

	// TODO implement?
	const handleClick = (stack_id) => {
		console.log('handleClick ' + stack_id);
		setStack(stacks[stack_id]);
	};

	// TODO: Font sizes are a mess right now. Find some way to gracefully select
	// them and pass them through the hierarchy.
	const fontSize = '2.5px';
	const fontColor = 'black';

	const stack_breadcrumb = () => {
		if (!stack) return;

		return (
			<Typography
        sx={{ display: 'flex', alignItems: 'center', color: fontColor }}
        fontSize={fontSize}
        color="text.primary"
      >
        <LayersIcon sx={{ ml: 0.1, mr: 0.1 }} fontSize={fontSize} />
        {stack.image_scale.toFixed(2) + '"/px ' + stack.filter}
      </Typography>
	   );
	};

	const mosaic_or_stack_panel = () => {
		if (stack) {
			return (
				<StackPanel
					stack_id={stack.id}
					stacks={stacks}
					handleClick={handleClick}
				/>
			);
		} else if (mosaic_id) {
			return (
				<MosaicPanel
					mosaic_id={mosaic_id}
					mosaics={mosaics}
					targets={targets}
					stacks={stacks}
					handleClick={handleClick}
				/>
			);
		}
	};

	const mosaic_breadcrumb_clicked = (event) => {
		console.log('mosaic breadcrumb clicked');
		event.stopPropagation();
	};

	return (
    <div role="presentation">
      <Breadcrumbs
      	aria-label="breadcrumb"
      	fontSize={fontSize}
      	sx={{
      		color: 'black', ml: 0.15, mt: 0.2, mb: 0
      	}}>
        <Link
          underline="hover"
          sx={{ display: 'flex', alignItems: 'center', color: fontColor }}
          style={{pointerEvents: "auto"}}
          fontSize={fontSize}
          color="inherit"
          onClick={(event) => {
				    setStack(undefined);
				    event.stopPropagation();
				  }}
        >
          <AutoAwesomeMosaicIcon sx={{ ml: 0.1, mr: 0.1 }} fontSize={fontSize} />
          {mosaic.name} Mosaic
        </Link>
        {stack_breadcrumb()}
      </Breadcrumbs>
      {mosaic_or_stack_panel()}
    </div>
  );
}
