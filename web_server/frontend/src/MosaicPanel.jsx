import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import MosaicOutline from "./MosaicOutline.jsx";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

// TODO: Add "mosaic not found" messages.
export default function MosaicPanel({ stacks, targets, mosaics }) {
	const params = useParams();
	const navigate = useNavigate();

	const mosaic = mosaics[params.mosaic_id];

	const fontColor = "#aaa";

	const get_filters = () => {
	  let filters = new Set();
	  for (let target_id of mosaic.target_ids) {
	    for (let stack_id of targets[target_id].stack_ids) {
	      filters.add(stacks[stack_id].filter);
	    }
	  }

	  const filter_array = Array.from(filters);
	  console.log('got filters', filter_array);
		return filter_array;
	}

	return (
		<>
			<Breadcrumbs aria-label="breadcrumb">
	      <Link
	      	href="/"
	        underline="hover"
	        sx={{
	        	display: 'flex',
	        	alignItems: 'center',
	        	color: fontColor,
	        	marginLeft: '0.5em',
	        	marginRight: '0.5em',
	        }}
	        style={{pointerEvents: "auto"}}
	        color="inherit"
	      >
	        <AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }}/>
	        Starmap
	      </Link>
	      <Typography color="text.primary" margin="0.5em">
	      	{mosaic.name} Mosaic
	      </Typography>
	    </Breadcrumbs>

			<div id="mosaicPanel">
				<MosaicOutline
          stacks={stacks}
          targets={targets}
          mosaics={mosaics}
          mosaic_id={params.mosaic_id}
        />
			</div>
			<div style={{
				  marginLeft: "1em",
				  marginRight: "1em",
		    	display: "grid",
		    	gridTemplateColumns: "repeat(4, 1fr)",
		    	gridColumnGap: "1em",
		    	gridRowGap: "1em",
		  }}>
				{get_filters().map(filter => {
					console.log('filter', filter);
					
				  return <img
            src={`/mosaic_preview_image_for_filter/${params.mosaic_id}/${filter}`}
            style={{
              height: "100%",
              maxWidth: "100%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />;
				})}
			</div>
		</>
	);
}
