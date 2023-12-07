import React, { useState, useRef, useEffect, useReducer } from "react";
import { useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function MosaicPanel({ mosaics, targets, stacks, handleClick }) {
	let params = useParams();

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
	      	{mosaics[params.mosaic_id].name} Mosaic
	      </Typography>
	    </Breadcrumbs>

			<div id="mosaicPanel" style={{margin: "2px"}}>
				<span>mosaic panel {params.mosaic_id}</span>
			</div>
		</>
	);
}

// <table width="100%">
// 	<thead height="0"></thead>
// 	<tbody>
// 		{mosaics[params.mosaic_id].target_ids.map((target_id) => (
// 			<TargetPanel
// 			  id={"TargetPanel_" + target_id}
// 			  key={target_id}
// 				target={targets[target_id]}
// 				stacks={stacks}
// 				handleClick={handleClick}
// 			/>
// 		))}
// 	</tbody>
// </table>