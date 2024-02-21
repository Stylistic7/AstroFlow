import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';

import MosaicOutline from "./MosaicOutline.jsx";
import MultiChannelImage from "./MultiChannelImage.jsx";
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

	// TODO: Add other modes besides AUTO.
	const runAllTasks = () => {
		for (let target_id of mosaic.target_ids) {
			for (let stack_id of targets[target_id].stack_ids) {
				fetch(`/add_to_run_cart/auto/stack_subframes/${stack_id}`);
			}
		}
	};


	// TODO: Add a dialog box to the "run" button that shows the user what will be
	// run, and how long it will likely take, and allow them to accept it.
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
					style={{ pointerEvents: "auto" }}
					color="inherit"
				>
					<AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }} />
					Starmap
				</Link>
				<Typography color="text.primary" margin="0.5em">
					{mosaic.name} Mosaic
				</Typography>
			</Breadcrumbs>

			<Typography color="text.primary" margin="0.5em" onClick={runAllTasks}>
				<DirectionsRunIcon sx={{ marginRight: '0.5em' }} onClick={runAllTasks} /> Run all tasks
			</Typography>
			<MosaicOutline
				stacks={stacks}
				targets={targets}
				mosaics={mosaics}
				mosaic_id={params.mosaic_id}
			/>
			<MultiChannelImage
				stacks={stacks}
				targets={targets}
				mosaics={mosaics}
				mosaic_id={params.mosaic_id}
			/>
		</>
	);
}
