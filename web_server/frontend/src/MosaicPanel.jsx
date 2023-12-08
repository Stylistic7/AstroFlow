import React, { useState, useRef, useEffect, useReducer } from "react";
import { useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import Polyline from "./Polyline.jsx";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function MosaicPanel({ mosaics, targets, stacks, handleClick }) {
	let params = useParams();
	let mosaic = mosaics[params.mosaic_id];

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
							className="target_polygon"
							polyline={targets[target_id].footprint}
							key={target_id}
						/>
					)}
					<Polyline
						className="mosaic_polygon"
						polyline={mosaic.footprint}
						key={params.mosaic_id}
					/>
				</svg>
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