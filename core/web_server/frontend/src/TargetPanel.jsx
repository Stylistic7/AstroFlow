import React, { useContext } from "react";
import { useParams, useLocation } from "react-router-dom";

import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

import StackThumbnail from "./StackThumbnail.jsx";
import QuarantineToggle from './QuarantineToggle.jsx';
import "./Starmap.css";
import AstroFlowContext from './AstroFlowContext.jsx';


export default function TargetPanel() {
	const { stacks, targets } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();

	const target = targets[params.target_id];

	return (
		<>
			<Box sx={{ display: 'flex', alignItems: 'center', ml: 2, mb: 1 }}>
				<Typography variant="h6" sx={{ mr: 1 }}>
					{target?.name || params.target_id}
				</Typography>
				<QuarantineToggle
					objectType="target"
					objectId={params.target_id}
					isQuarantined={target?.is_quarantined || false}
				/>
			</Box>
			<div style={{
				display: "grid",
				gridTemplateColumns: "repeat(4, 1fr)",
				maxHeight: "80vh",
				gap: "1em",
				marginLeft: "2em",
				marginRight: "2em",
				opacity: target?.is_quarantined ? 0.5 : 1,
			}}>
				{target?.stack_ids?.map((stack_id) => (
					<StackThumbnail
						key={"StackThumbnail_" + stack_id}
						stacks={stacks}
						stack_id={stack_id}
						handleClick={event => window.open(`${location.pathname}/stack/${stack_id}`, '_blank')}
					/>
				))}
			</div>
		</>
	);
}
