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

	console.log('viewbox', mosaic.viewbox.join(" "))

	const fontColor = "#aaa";

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
		</>
	);
}
