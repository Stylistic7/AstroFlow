import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useContext } from 'react';

import DeleteIcon from '@mui/icons-material/Delete';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

import { trackPromise, usePromiseTracker } from "react-promise-tracker";
import ClipLoader from "react-spinners/ClipLoader";

import Typography from '@mui/material/Typography';

import StackThumbnail from "./StackThumbnail.jsx";
import RunPanel from "./RunPanel.jsx";
import "./Starmap.css";
import AstroFlowContext from './AstroFlowContext.jsx';
import DivWithActions from './DivWithActions.jsx';
import StackActionButtons from './StackActionButtons.jsx';
import { ra_to_hms, dec_to_dms } from './units.js'
import TileViewer from './TileViewer.jsx';

function QuarantineStackDialog({ stack_id }) {
	const [open, setOpen] = React.useState(false);

	const handleClickOpen = () => {
		setOpen(true);
	};

	const handleDisagree = () => {
		setOpen(false);
	};

	const handleAgree = () => {
		console.log('Quarantining stack id ' + stack_id);
		fetch(`/quarantine_stack/${stack_id}`);
		setOpen(false);
	};

	const handleClose = () => {
		setOpen(false);
	};

	// TODO: Put the correct number of light frames into the text of the dialog.
	//
	// TODO: Show a spinner until the quarantine and re-run is done, then maybe
	// go up a level in breadcrumbs.
	return (
		<React.Fragment>
			<Typography color="text.primary" margin="0.5em" sx={{ cursor: 'pointer' }} onClick={handleClickOpen}>
				<DeleteIcon sx={{ marginRight: '0.5em', cursor: 'pointer' }} onClick={handleClickOpen} /> Quarantine light frames
			</Typography>
			<Dialog
				open={open}
				onClose={handleClose}
				aria-labelledby="alert-dialog-title"
				aria-describedby="alert-dialog-description"
			>
				<DialogTitle id="alert-dialog-title">
					{"Quarantine this stack's light frames?"}
				</DialogTitle>
				<DialogContent>
					<DialogContentText id="alert-dialog-description">
						<p>
							This will quarantine light frames, but will not affect dark, bias,
							or flat frames.
						</p>
						<p>
							The quarantined frames will still be on your disk, and will not
							be moved. AstroFlow will simply note them as quarantined. You are
							responsible for deleting them, if you desire.
						</p>
						<p>
							This action will eliminate this stack from AstroFlow.
						</p>
					</DialogContentText>
				</DialogContent>
				<DialogActions>
					<Button onClick={handleDisagree}>Disagree</Button>
					<Button onClick={handleAgree} autoFocus>Agree</Button>
				</DialogActions>
			</Dialog>
		</React.Fragment>
	);
}

const override: CSSProperties = {
	display: "block",
	margin: "0 auto",
	borderColor: "red",
};

// TODO: The StackPanel should contain the following information:
// 
// - status
//   - {not stacked, stacked, post-processed}
// 
// - stack health
//   - sanity check results
// 
// - visualization of data quality (e.g. star count, clouds, etc.)
//   - video
//   - plot
// 
// - results
//   - stacked master
//   - post-processed
//
// and the following actions:
//
// - quarantine or delete the entire stack
// - quarantine or delete just the poor images in the stack
// - select flat frames
// - queue the stack for stacking
// - configure stacking options and post-processing scripts
export default function StackPanel() {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	const params = useParams();
	const mosaic = mosaics[params.mosaic_id];
	const target = targets[params.target_id];
	const stack = stacks[params.stack_id];

	console.log('mosaic_id', params.mosaic_id,
		'target_id', params.target_id,
		'stack_id', params.stack_id
	);

	const plotRef = useRef();
	const { promiseInProgress } = usePromiseTracker();

	let [color, setColor] = useState("#ffffff");
	const [open, setOpen] = React.useState(false);

	const [flat_stack_ids, set_flat_stack_ids] = useState([]);

	const handleClickOpen = () => {
		setOpen(true);
	};

	const handleClose = () => {
		setOpen(false);
	};

	// TODO: Don't hardcode the font color; use the theme.
	const fontColor = "#aaa";

	// const fetchHtml = (stack_id) => {
	// 	console.log('fetchHtml');

	// fetch(`/get_num_stars_histogram_html/${stack_id}`)
	// 	.then((response) => {
	// 		return response.text();
	// 	})
	// 	.then((htmlString) => {
	// 		if (plotRef.current) {
	// 			if (plotRef.current.children[0]) {
	// 				plotRef.current.removeChild(plotRef.current.children[0]);
	// 			}

	// 			const fragment = document.createRange().createContextualFragment(htmlString);
	// 			plotRef.current.append(fragment);
	// 		}
	// 	});

	// 	fetch(`/stacks_for_type_and_camera/flat/${stack.camera}`)
	// }

	useEffect(() => {
		trackPromise(
			fetch(`/flats_for_camera_and_filter/${stack.camera}/${stack.filter}`)
				.then(r => r.json())
				.then(j => set_flat_stack_ids(j))
		);
	}, []);

	let stack_thumbnail_video_url = `/stack_subframe_thumbnail_video/${params.stack_id}`;

	console.log('promiseInProgress', promiseInProgress);

	return (
		<>
			<div style={{ margin: "2px" }}>
				<div>
					<Typography color="text.primary" margin="0.5em">
						Stack contains {stack.contents.length} runs of light frames
					</Typography>
					<Typography color="text.primary" margin="0.5em">
						RA {ra_to_hms(stack.center[0])}, Dec {dec_to_dms(stack.center[1])}
					</Typography>
					<QuarantineStackDialog
						stack_id={params.stack_id}
						open={open}
						onClose={handleClose}
					/>
				</div>
				<div style={{ maxWidth: "50%" }}>
					<StackThumbnail
						key={"StackThumbnail_" + params.stack_id}
						stacks={stacks}
						stack_id={params.stack_id}
					/>
				</div>
				<Box sx={{ width: '75vw', height: '75vh', border: '1px solid #444' }}>
					<TileViewer imageHash={params.stack_id} />
				</Box>
				{/* <div ref={plotRef} style={{ maxWidth: "30%" }}></div> */}

				{promiseInProgress ? (
					<ClipLoader
						color={color}
						loading={promiseInProgress}
						cssOverride={override}
						size={150}
						aria-label="Loading Spinner"
						data-testid="loader"
					/>
				) : (
					Object.values(stack.contents).map(substack =>
						<div style={{ padding: '20px' }}>
							<DivWithActions
								title={`Run of ${substack.frames.length} light frames on ${stack.oldest_date}`}
								action_buttons={<StackActionButtons stack_id={substack.content_hash} />}
							>
								<RunPanel
									key={substack.content_hash}
									stack={substack}
									flat_stack_ids={flat_stack_ids}
								/>
							</DivWithActions>
						</div>
					)
				)}
			</div>
		</>
	);
}
