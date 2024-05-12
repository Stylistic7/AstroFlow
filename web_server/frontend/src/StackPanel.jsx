import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import LayersIcon from '@mui/icons-material/Layers';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import DeleteIcon from '@mui/icons-material/Delete';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

import { trackPromise, usePromiseTracker } from "react-promise-tracker";
import ClipLoader from "react-spinners/ClipLoader";

import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import HDRImage from "./HDRImage.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

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
			<Typography color="text.primary" margin="0.5em" onClick={handleClickOpen}>
				<DeleteIcon sx={{ marginRight: '0.5em' }} onClick={handleClickOpen} /> Quarantine light frames
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
export default function StackPanel({ stacks, targets, mosaics }) {
	const params = useParams();
	const stack_id = params.stack_id;
	const stack = stacks[stack_id];
	const plotRef = useRef();
	const { promiseInProgress } = usePromiseTracker();

	let [color, setColor] = useState("#ffffff");
	const [open, setOpen] = React.useState(false);

	const [flatRuns, setFlatRuns] = useState();

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
			fetch(`/stacks_for_type_and_camera/flat/${stack.camera}`)
				.then(r => r.json())
				.then(j => setFlatRuns(j))
		);
	}, []);

	let stack_thumbnail_video_url = `/stack_thumbnail_video/${stack_id}`;

	console.log('promiseInProgress', promiseInProgress);

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
					{stack.target} Stack
				</Typography>
			</Breadcrumbs>

			<div style={{ margin: "2px" }}>
				<div>
					<Typography color="text.primary" margin="0.5em">
						Stack contains {stack.runs.length} runs of light frames
					</Typography>
					<QuarantineStackDialog
						stack_id={stack_id}
						open={open}
						onClose={handleClose}
					/>
				</div>
				<div style={{ maxWidth: "50%" }}>
					<StackThumbnail
						key={"StackThumbnail_" + stack_id}
						stacks={stacks}
						stack_id={stack_id}
					/>
				</div>
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
					<span>{flatRuns}</span>
				)}
			</div>
		</>
	);
}
