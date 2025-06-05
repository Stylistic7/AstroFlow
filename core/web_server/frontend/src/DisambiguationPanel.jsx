import React, { useContext } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AstroFlowContext from './AstroFlowContext.jsx';
import StackCard from "./StackCard.jsx";
import TargetCard from "./TargetCard.jsx";
import MosaicCard from "./MosaicCard.jsx";
import CalibrationStackCard from "./CalibrationStackCard.jsx";

export default function DisambiguationPanel({ items_in }) {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	// const params = useParams();
	let { items } = useParams();
	items = items || items_in;
	console.log('DisambiguationPanel items', items);

	if (!items || items.trim() === "") {
		return (
			<div style={{
				margin: "2em",
				padding: "2em",
				textAlign: "center",
				color: "#999",
				fontStyle: "italic",
				fontSize: "1.25em",
			}}>
				No results found.
			</div>
		);
	}

	return (
		<>
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
					if (type == "stack") {
						return <StackCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							stack_id={id}
						/>;
					} else if (type == "target") {
						return <TargetCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							target_id={id}
						/>;
					} else if (type == "mosaic") {
						return <MosaicCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							mosaic_id={id}
						/>;
					} else if (type == "calibration_stack") {
						return <CalibrationStackCard
							key={item}
							stacks={stacks}
							targets={targets}
							mosaics={mosaics}
							stack_id={id}
						/>;
					}
				}
				)}
			</div>
		</>
	);
}


