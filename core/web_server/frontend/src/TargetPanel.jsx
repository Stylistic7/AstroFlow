import React, { useContext } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import DeleteIcon from '@mui/icons-material/Delete';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';

import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";
import AstroFlowContext from './AstroFlowContext.jsx';

function QuarantineTargetDialog({ target_id }) {
	const [open, setOpen] = React.useState(false);

	const handleClickOpen = () => {
		setOpen(true);
	};

	const handleDisagree = () => {
		setOpen(false);
	};

	const handleAgree = () => {
		console.log('Quarantining target id ' + target_id);
		fetch(`/quarantine_target/${target_id}`);
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
					{"Quarantine this target's light frames?"}
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
							This action will eliminate this target from AstroFlow.
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

export default function TargetPanel() {
	const { stacks, targets } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();
	console.log('target_id', params.target_id);

	const target = targets[params.target_id];
	console.log('target', target);

	const navigate = useNavigate();

	const [open, setOpen] = React.useState(false);

	const handleClickOpen = () => {
		setOpen(true);
	};

	const handleClose = () => {
		setOpen(false);
	};


	// TODO: Don't hardcode the font color; use the theme.
	const fontColor = "#aaa";

	return (
		<>
			<QuarantineTargetDialog
				target_id={params.target_id}
				open={open}
				onClose={handleClose}
			/>
			<div style={{
				display: "grid",
				gridTemplateColumns: "repeat(4, 1fr)",
				maxHeight: "80vh",
				gap: "1em",
				marginLeft: "2em",
				marginRight: "2em",
			}}>
				{target.stack_ids.map((stack_id) => (
					<StackThumbnail
						key={"StackThumbnail_" + stack_id}
						stacks={stacks}
						stack_id={stack_id}
						handleClick={event => navigate(`${location.pathname}/stack/${stack_id}`)}
					/>
				))}
			</div>
		</>
	);
}


