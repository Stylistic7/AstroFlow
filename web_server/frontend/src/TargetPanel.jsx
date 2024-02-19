import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

export default function TargetPanel({ stacks, targets, mosaics }) {
	const params = useParams();
	console.log('target_id', params.target_id);

	const target = targets[params.target_id];

	console.log('target', target);

	const navigate = useNavigate();

	// TODO: Don't hardcode the font color; use the theme.
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
					style={{ pointerEvents: "auto" }}
					color="inherit"
				>
					<AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }} />
					Starmap
				</Link>
				<Typography color="text.primary" margin="0.5em">
					{target.name} Target
				</Typography>
			</Breadcrumbs>
			<div style={{
				display: "grid",
				gridTemplateColumns: "repeat(4, 1fr)",
				height: "80vh",
				gap: "1em",
				marginLeft: "2em",
				marginRight: "2em",
			}}>
				{target.stack_ids.map((stack_id) => (
					<StackThumbnail
						key={"StackThumbnail_" + stack_id}
						stack={stacks[stack_id]}
						handleClick={event => navigate(`/stack/${stack_id}`)}
					/>
				))}
			</div>
		</>
	);
}


