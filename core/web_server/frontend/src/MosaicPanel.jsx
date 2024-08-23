import React, { useState, useRef, useEffect, useReducer, useContext } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";

import Typography from '@mui/material/Typography';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import Snackbar from '@mui/material/Snackbar';

import AstroFlowBreadcrumbs from "./AstroFlowBreadcrumbs.jsx";
import MosaicOutline from "./MosaicOutline.jsx";
import ColorChannelSelector from "./ColorChannelSelector.jsx";
import TargetPanel from "./TargetPanel.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

import "./Starmap.css";

// TODO: Add "mosaic not found" messages.
export default function MosaicPanel() {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();
	const navigate = useNavigate();

	const mosaic = mosaics[params.mosaic_id];

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

	const [filter, setFilter] = useState(get_filters()[0]);
	console.log("filter is", filter);

	const fontColor = "#aaa";

	const handleSetFilter = f => {
		console.log('handleSetFilter', f);
		setFilter(f);
	};

	const divmod = (y, x) => {
		let div = Math.trunc(y / x);
		let rem = y % x;

		return [div, rem];
	}

	const ra_to_hms = (ra) => {
		let [hours, hours_remainder] = divmod(ra * 3600, 15 * 3600);
		let [minutes, minutes_remainder] = divmod(hours_remainder, 15 * 60);
		let seconds = Math.round(minutes_remainder / 15);

		return `${String(hours).padStart(2, '0')}hr ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`;
	}

	const dec_to_dms = (dec) => {
		let negative = dec < 0 ? '-' : '';
		dec = Math.abs(dec);
		let [degrees, degrees_remainder] = divmod(dec * 3600, 3600);
		degrees_remainder = Math.abs(degrees_remainder);
		let [minutes, seconds] = divmod(degrees_remainder, 60);
		seconds = Math.round(seconds);

		return `${negative}${String(degrees).padStart(2, '0')}º ${String(minutes).padStart(2, '0')}' ${String(seconds).padStart(2, '0')}"`
	}

	const telescopius_pane_line = (mosaic, target_id, pane) => {
		let [tile_x, tile_y] = mosaic.tile_indices[pane];
		let target = targets[target_id];
		let stack_id = target.stack_ids[0];
		let stack = stacks[stack_id];
		let [ra, dec] = stack.center;
		let rotation_angle = stack.rotation_angle * 180.0 / Math.PI;
		let pane_width = stack.image_scale * stack.width_pixels / 60.0;
		let pane_height = stack.image_scale * stack.height_pixels / 60.0;

		return `Pane ${pane + 1}, ${ra_to_hms(ra)}, ${dec_to_dms(dec)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(2)}, ${pane_height.toFixed(2)}, 20%, ${tile_x + 1}, ${tile_y + 1}`;
	};

	const [open, setOpen] = useState(false);

	const telescopius = mosaic => {
		const header = "Pane, RA, DEC, Position Angle (East), Pane width (arcmins), Pane height (arcmins), Overlap, Row, Column";
		const text = [header].concat(
			mosaic.target_ids.map((target_id, pane) =>
				telescopius_pane_line(mosaic, target_id, pane)
			)).join('\n');
		console.log('text', text);

		return <>
			<p>
				{header}<br />
				{mosaic.target_ids.map((target_id, pane) => <>
					<span>
						{telescopius_pane_line(mosaic, target_id, pane)}
					</span><br />
				</>)}
			</p >
			<CopyToClipboard text={text}
				onCopy={() => setOpen(true)}>
				<button>Copy to clipboard with button</button>
			</CopyToClipboard>
		</>;
	};

	// TODO: Add other modes besides AUTO.
	const runAllTasks = () => {
		fetch(`/add_mosaic_to_shopping_cart/${params.mosaic_id}`);
	};

	const imageType = "mosaic_thumbnail_image_for_filter";

	// TODO: Add a dialog box to the "run" button that shows the user what will be
	// run, and how long it will likely take, and allow them to accept it.
	return (
		<>
			<AstroFlowBreadcrumbs />
			<Accordion>
				<AccordionSummary
					expandIcon={<ExpandMoreIcon />}
					aria-controls="telescopius-content"
					id="telescopius-header"
				>
					Telescopius
				</AccordionSummary>
				<AccordionDetails>
					{telescopius(mosaic)}
				</AccordionDetails>
			</Accordion>
			<Snackbar
				anchorOrigin={{ vertical: "top", horizontal: "center" }}
				open={open}
				autoHideDuration={5000}
				onClose={event => setOpen(false)}
				message="Telescopius text copied"
				key="telescopius-snackbar"
			/>

			<Typography color="text.primary" margin="0.5em" onClick={runAllTasks}>
				<DirectionsRunIcon sx={{ marginRight: '0.5em' }} onClick={runAllTasks} /> Run all tasks
			</Typography>

			<div id="mosaicPanel"
				style={{
					display: "block",
					margin: "1em",
					position: "relative",
					maxWidth: "75vw",
					maxHeight: "75vh",
					marginLeft: "auto",
					marginRight: "auto"
				}}>
				<img
					src={`/${imageType}/${params.mosaic_id}/${filter}`}
					style={{
						display: "block",
						marginLeft: "auto",
						marginRight: "auto",
						maxWidth: "75vw",
						maxHeight: "75vh"
					}}
				/>
				<MosaicOutline
					stacks={stacks}
					targets={targets}
					mosaics={mosaics}
					mosaic_id={params.mosaic_id}
					filter={filter}
					style={{
						display: "block",
						position: "absolute",
						left: 0,
						top: 0,
						width: "100%",
						height: "100%",
						marginLeft: "auto",
						marginRight: "auto",
						maxWidth: "auto",
						maxHeight: "auto",
					}}
				/>
			</div>
			<ColorChannelSelector
				stacks={stacks}
				targets={targets}
				mosaics={mosaics}
				mosaic_id={params.mosaic_id}
				filter={filter}
				setFilter={handleSetFilter}
				style={{
					height: "100%",
					maxWidth: "100%",
					display: "block",
					marginLeft: "auto",
					marginRight: "auto",
				}}
			/>
		</>
	);
}
