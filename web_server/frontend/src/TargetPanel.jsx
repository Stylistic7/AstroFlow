import React, { useState, useRef, useEffect, useReducer } from "react";
import Grid from '@mui/material/Unstable_Grid2';
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

export default function TargetPanel({ target, stacks, zoomToElement }) {
	return (
		<div>
			<Grid container spacing={2} height="20px">
				{target.stack_ids.map((stack_id) => (
					<Grid xs={1}>
						<StackThumbnail
							stack={stacks[stack_id]}
							zoomToElement={zoomToElement}
						/>
					</Grid>
				))}
			</Grid>
		</div>
	);
}
