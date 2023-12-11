import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import MosaicCard from "./MosaicCard.jsx";
import TargetCard from "./TargetCard.jsx";

export default function DisambiguationPanel({ stacks, targets, mosaics }) {
	const params = useParams();

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
	        style={{pointerEvents: "auto"}}
	        color="inherit"
	      >
	        <AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }}/>
	        Starmap
	      </Link>
	      <Typography color="text.primary" margin="0.5em">
	      	Disambiguation
	      </Typography>
	    </Breadcrumbs>

			<div style={{
		    	display: "grid",
		    	gridTemplateColumns: "repeat(4, 1fr)",
		    	border: "1px solid red",
		    	height: "80vh",
		    	width: "100%"
		  }}>
				{params.items.split(";").map(item =>
					{
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

	
