import React, { useContext } from "react";
import { useParams, useLocation } from "react-router-dom";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import QuarantineToggle from "./QuarantineToggle.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';


export default function TargetPanel() {
	const { stacks, targets, isQuarantined } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();

	const target = targets[params.target_id];

	return (
		<div>
			<div style={{ display: "flex", alignItems: "center", marginLeft: "2em", marginBottom: "0.5em" }}>
				<QuarantineToggle objectType="target" objectId={params.target_id} />
			</div>
			<div style={{
					display: "grid",
					gridTemplateColumns: "repeat(4, 1fr)",
					maxHeight: "80vh",
					gap: "1em",
					marginLeft: "2em",
					marginRight: "2em",
					opacity: isQuarantined("target", params.target_id) ? 0.5 : 1,
				}}>
					{target?.stack_ids?.map((stack_id) => (
						<StackThumbnail
							key={"cell_" + stack_id}
							stacks={stacks}
							stack_id={stack_id}
							handleClick={event => window.open(`${location.pathname}/stack/${stack_id}`, '_blank')}
						>
							<DecisionBadge objectType="stack" objectId={stack_id} />
						</StackThumbnail>
					))}
			</div>
		</div>
	);
}
