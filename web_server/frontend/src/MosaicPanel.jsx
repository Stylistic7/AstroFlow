import React, { useState, useRef, useEffect, useReducer, useContext } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';

import MosaicOutline from "./MosaicOutline.jsx";
import ColorChannelSelector from "./ColorChannelSelector.jsx";
import TargetPanel from "./TargetPanel.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

import "./Starmap.css";

// TODO: Add "mosaic not found" messages.
export default function MosaicPanel() {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	const params = useParams();
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

	const telescopius = (mosaic) => {
		let output = [<p>Pane, RA, DEC, Position Angle (East), Pane width (arcmins), Pane height (arcmins), Overlap, Row, Column</p>];
		let pane = 1;
		for (let i = 0; i < mosaic.target_ids.length; ++i) {
			let target_id = mosaic.target_ids[i];
			let [tile_x, tile_y] = mosaic.tile_indices[i];

			let target = targets[target_id];
			let stack_id = target.stack_ids[0];
			let stack = stacks[stack_id];

			let [ra, dec] = stack.center;
			let rotation_angle = stack.rotation_angle * 180.0 / Math.PI;
			console.log('stack', stack);
			let pane_width = stack.image_scale * stack.width_pixels / 60.0;
			let pane_height = stack.image_scale * stack.height_pixels / 60.0;

			output.push(<p>{`Pane ${pane}, ${ra_to_hms(ra)}, ${dec_to_dms(dec)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(2)}, ${pane_height.toFixed(2)}, 20%, ${tile_x + 1}, ${tile_y + 1}`}</p>);
			pane += 1;
		}

		return output;
	}

	// TODO: Add other modes besides AUTO.
	const runAllTasks = () => {
		fetch(`/add_mosaic_to_shopping_cart/${params.mosaic_id}`);
	};

	const imageType = "mosaic_thumbnail_image_for_filter";

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
				{telescopius(mosaic)}
			</Typography>

			<Typography color="text.primary" margin="0.5em" onClick={runAllTasks}>
				<DirectionsRunIcon sx={{ marginRight: '0.5em' }} onClick={runAllTasks} /> Run all tasks
			</Typography>

			<div id="mosaicPanel"
				style={{
					display: "block",
					margin: "1em",
					position: "relative",
				}}>
				<img
					src={`/${imageType}/${params.mosaic_id}/${filter}`}
					style={{
						height: "100%",
						maxWidth: "100%",
						display: "block",
						marginLeft: "auto",
						marginRight: "auto",
					}}
				/>
				<MosaicOutline
					stacks={stacks}
					targets={targets}
					mosaics={mosaics}
					mosaic_id={params.mosaic_id}
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
