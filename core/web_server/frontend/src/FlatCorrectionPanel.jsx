import React, { useState, useRef, useEffect, useReducer } from "react";
import { useParams } from "react-router-dom";
import { useContext } from 'react';

import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";
import AstroFlowContext from './AstroFlowContext.jsx';


export default function FlatCorrectionPanel({ handleClick }) {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	let params = useParams();
	const [trialFlatCorrections, setTrialFlatCorrections] = useState([]);

	const fontColor = "#aaa";

	useEffect(() => {
		fetch(`/trial_flat_corrections/${params.stack_id}`)
			.then(res => res.json()).then(data => {
				setTrialFlatCorrections(data);
				console.log(data);
			})
	}, []);

	return (
		<div id="flatCorrectionImages" style={{ margin: "2px" }}>
			{trialFlatCorrections.map((e) =>
				<div key={e.light_run + '_' + e.flat_run}>
					<img src={e.original_image} width="200" />
					<span>&nbsp;</span>
					{e.trial_flat_corrections.map((t) =>
						<img
							src={t.corrected_image}
							width="200"
							style={{
								margin: "2px",
								border: "1px solid red"
							}}
						/>
					)}
				</div>
			)}
		</div>
	);
}

