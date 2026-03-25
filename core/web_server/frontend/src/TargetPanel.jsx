import React, { useContext } from "react";
import { useParams, useLocation } from "react-router-dom";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';


export default function TargetPanel() {
	const { stacks, targets } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();

	const target = targets[params.target_id];

	return (
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
					<div key={"cell_" + stack_id} style={{ position: "relative" }}>
						<StackThumbnail
							stacks={stacks}
							stack_id={stack_id}
							handleClick={event => window.open(`${location.pathname}/stack/${stack_id}`, '_blank')}
						/>
						<div style={{ position: "absolute", top: 0, right: 0, zIndex: 1 }}>
							<DecisionBadge objectType="stack" objectId={stack_id} />
						</div>
					</div>
				))}
		</div>
	);
}
