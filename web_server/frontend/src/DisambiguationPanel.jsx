import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import MosaicCard from "./MosaicCard.jsx";
import TargetCard from "./TargetCard.jsx";

export default function DisambiguationPanel({ stacks, targets, mosaics, items_in }) {
	// const params = useParams();
	let { items } = useParams();
	items = items || items_in;
	console.log('DisambiguationPanel items', items);

	return (
		<>
			<Breadcrumbs aria-label="breadcrumb">
				<Link
					href="/"
					underline="hover"
					sx={{
						display: 'flex',
						alignItems: 'center',
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
					Disambiguation
				</Typography>
			</Breadcrumbs>

			<div style={{
				marginLeft: "1em",
				marginRight: "1em",
				display: "grid",
				gridTemplateColumns: "repeat(4, 1fr)",
				gridColumnGap: "1em",
				gridRowGap: "1em",
			}}>
				{items.split(";").map(item => {
					let [type, id] = item.split(",");
					if (type == "mosaic") {
						return <MosaicCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							mosaic_id={id}
						/>;
					} else if (type == "target") {
						return <TargetCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							target_id={id}
						/>;
					}
				}
				)}
			</div>
		</>
	);
}


